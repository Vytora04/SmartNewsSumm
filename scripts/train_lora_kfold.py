# scripts/train_lora_kfold.py
"""
LoRA fine-tuning with K-Fold cross validation on the local BBC News Summary dataset.

Dataset structure (already in your repo zip):
  data/bbc-news-summary/news/<category>/*.txt
  data/bbc-news-summary/summaries/<category>/*.txt

Run (from project root):
  python -m scripts.train_lora_kfold

Outputs:
  results/lora_kfold/
    fold_0/adapter/  (LoRA adapter)
    fold_0/metrics.json
    ...
    kfold_summary.json
"""

import os
import json
import time
import random
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import KFold

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model, TaskType

from backend.metrics import evaluate_summary

BASE_MODEL = os.environ.get("BASE_MODEL", "sshleifer/distilbart-cnn-12-6")

BBC_ROOT = os.environ.get("BBC_ROOT", "data/bbc-news-summary")
OUT_ROOT = os.environ.get("OUT_ROOT", "results/lora_kfold")

K_FOLDS = int(os.environ.get("K_FOLDS", "5"))
SEED = int(os.environ.get("SEED", "42"))

MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", "512"))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", "128"))

EPOCHS = float(os.environ.get("EPOCHS", "1"))
LR = float(os.environ.get("LR", "2e-4"))

TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH", "1"))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))

EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "100"))


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bbc_dataset(root_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load BBC News Summary dataset from local folder.
    Expects:
      root/news/<category>/*.txt
      root/summaries/<category>/*.txt
    """
    root = Path(root_dir)
    news_dir = root / "news"
    summaries_dir = root / "summaries"

    if not news_dir.exists() or not summaries_dir.exists():
        raise FileNotFoundError(
            f"BBC dataset not found at: {root.resolve()}\n"
            f"Expected: {news_dir} and {summaries_dir}"
        )

    samples: List[Dict[str, Any]] = []
    for category_dir in news_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for article_path in category_dir.glob("*.txt"):
            summary_path = summaries_dir / category / article_path.name
            if not summary_path.exists():
                continue

            article = article_path.read_text(encoding="utf-8", errors="ignore").strip()
            reference = summary_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not article or not reference:
                continue

            samples.append(
                {
                    "id": article_path.stem,
                    "category": category,
                    "article": article,
                    "reference": reference,
                }
            )

            if limit and len(samples) >= limit:
                return samples

    return samples


def guess_lora_targets(model) -> List[str]:
    """
    BART-family models typically use q_proj/v_proj.
    We pick targets that actually exist in the model to avoid crashes.
    """
    preferred = ["q_proj", "v_proj", "k_proj", "out_proj"]
    found = set()
    for name, _ in model.named_modules():
        for p in preferred:
            if name.endswith(p):
                found.add(p)

    if "q_proj" in found and "v_proj" in found:
        return ["q_proj", "v_proj"]

    targets = [p for p in preferred if p in found]
    # Fallback: most BART-like models should still have q/v
    return targets or ["q_proj", "v_proj"]


class BBCSeq2SeqDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_src: int, max_tgt: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        model_inputs = self.tokenizer(
            item["article"],
            max_length=self.max_src,
            truncation=True,
        )
        # compatible with your repo's transformers usage
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                item["reference"],
                max_length=self.max_tgt,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


@torch.no_grad()
def eval_generated_metrics(model, tokenizer, samples: List[Dict[str, Any]], limit: int) -> Dict[str, float]:
    """
    Generate summaries for a subset of validation samples and compute ROUGE/BLEU
    using backend.metrics.evaluate_summary (same as your project eval).
    """
    model.eval()
    use_samples = samples[: min(limit, len(samples))]
    device = next(model.parameters()).device

    all_metrics: List[Dict[str, float]] = []
    for s in use_samples:
        inputs = tokenizer(
            s["article"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out_ids = model.generate(
            **inputs,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        m = evaluate_summary(
            generated=pred,
            reference=s["reference"],
            include_bertscore=False,
        )
        all_metrics.append(m)

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return avg


def train_one_fold(fold_idx: int, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    fold_dir = Path(OUT_ROOT) / f"fold_{fold_idx}"
    adapter_dir = fold_dir / "adapter"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    use_cuda = torch.cuda.is_available()
    use_fp16 = use_cuda

    USE_GRAD_CHECKPOINTING = bool(int(os.environ.get("GRAD_CHECKPOINTING", "0")))

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    if USE_GRAD_CHECKPOINTING:
        base.gradient_checkpointing_enable()

    targets = guess_lora_targets(base)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    
    model.config.use_cache = False

    if USE_GRAD_CHECKPOINTING:
        try:
            # Works on many HF models
            model.enable_input_require_grads()
        except Exception:
            # Robust fallback hook
            emb = model.get_input_embeddings()
            emb.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))

    model.train()

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters found. LoRA injection may have failed.")

    train_ds = BBCSeq2SeqDataset(train_samples, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN)
    val_ds = BBCSeq2SeqDataset(val_samples, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = dict(
        output_dir=str(fold_dir / "trainer_out"),
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        fp16=use_fp16,
        gradient_checkpointing=USE_GRAD_CHECKPOINTING,  # ✅ only if enabled
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=1,
        predict_with_generate=False,
        report_to="none",
    )

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        train_args["eval_strategy"] = "steps"
    else:
        train_args["evaluation_strategy"] = "steps"

    args = Seq2SeqTrainingArguments(**train_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,  # FutureWarning is fine for now
    )

    t0 = time.time()
    trainer.train()
    train_elapsed = time.time() - t0

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    metrics = eval_generated_metrics(model, tokenizer, val_samples, limit=EVAL_LIMIT)

    info = {
        "fold": fold_idx,
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "lr": LR,
        "max_source_len": MAX_SOURCE_LEN,
        "max_target_len": MAX_TARGET_LEN,
        "grad_accum": GRAD_ACCUM,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "eval_limit": min(EVAL_LIMIT, len(val_samples)),
        "lora_target_modules": targets,
        "train_elapsed_s": round(train_elapsed, 2),
        "metrics": metrics,
        "adapter_dir": str(adapter_dir).replace("\\", "/"),
        "used_cuda": use_cuda,
        "gradient_checkpointing": USE_GRAD_CHECKPOINTING,
    }

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\n[Fold {fold_idx}] metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return info



def main() -> None:
    set_seed(SEED)

    samples = load_bbc_dataset(BBC_ROOT)
    print(f"Loaded {len(samples)} BBC samples from {BBC_ROOT}")

    # Shuffle once (important for fold mix)
    rng = np.random.default_rng(SEED)
    rng.shuffle(samples)

    # Make sure output root exists
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_infos: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(samples)):
        fold_dir = out_root / f"fold_{fold_idx}"
        metrics_path = fold_dir / "metrics.json"
        adapter_dir = fold_dir / "adapter"

        # ---- RESUME / SKIP LOGIC (Option B) ----
        if metrics_path.exists() and adapter_dir.exists():
            # extra safety: ensure adapter dir has something inside
            has_any_adapter_files = any(adapter_dir.iterdir())
            if has_any_adapter_files:
                print(f"\n=== Fold {fold_idx+1}/{K_FOLDS} ===")
                print(f"Skipping fold {fold_idx} (already done): {metrics_path}")
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        fold_infos.append(json.load(f))
                    continue
                except Exception as e:
                    print(f"Warning: failed to load {metrics_path} ({e}). Re-training fold {fold_idx}.")

        # Otherwise train fold
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        print(f"\n=== Fold {fold_idx+1}/{K_FOLDS} ===")
        fold_infos.append(train_one_fold(fold_idx, train_samples, val_samples))

    # Aggregate
    if not fold_infos:
        raise RuntimeError("No fold results found (nothing trained and nothing loaded).")

    metric_keys = list(fold_infos[0].get("metrics", {}).keys())
    if not metric_keys:
        raise RuntimeError("No metrics found in fold_infos[0]. Check fold metrics.json format.")

    mean = {k: float(np.mean([fi["metrics"][k] for fi in fold_infos])) for k in metric_keys}
    std = {k: float(np.std([fi["metrics"][k] for fi in fold_infos])) for k in metric_keys}

    summary = {
        "base_model": BASE_MODEL,
        "bbc_root": BBC_ROOT,
        "k_folds": K_FOLDS,
        "epochs": EPOCHS,
        "lr": LR,
        "max_source_len": MAX_SOURCE_LEN,
        "max_target_len": MAX_TARGET_LEN,
        "eval_limit_per_fold": EVAL_LIMIT,
        "mean": mean,
        "std": std,
        "folds": fold_infos,
    }

    with open(out_root / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== K-Fold Summary (mean ± std) ===")
    for k in metric_keys:
        print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    print(f"\nSaved to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
