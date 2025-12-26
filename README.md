# 🚀 SmartNewsSumm

**SmartNewsSumm** is a state-of-the-art intelligent news summarization system. It utilizes a **Hybrid Extract-then-Abstract** architecture, supercharged with **LoRA (Low-Rank Adaptation)** and **RAG (Retrieval-Augmented Generation)** to deliver highly accurate, concise, and contextually rich summaries.

---

## ✨ Key Features

- **🏆 Hybrid Architecture**: Combines the precision of extractive methods (TextRank, TF-IDF) with the fluency of abstractive models (BART, DistilBART, T5).
- **🎨 Fine-tuned with LoRA**: Support for custom LoRA adapters to specialize the model on specific news domains (e.g., BBC dataset).
- **🔍 RAG Integration**: Context-aware retrieval system that injects relevant evidence directly into the generation process.
- **⚡ Performance Optimized**: Hybrid mode processes articles ~35% faster than standard transformer models by intelligently filtering input content.
- **🎯 Precise Length Control**: deterministic "word count" enforcement and percentage-based targets for summaries of any size.
- **🚫 Repetition Control**: Advanced n-gram suppression to ensure natural, non-repetitive summaries.
- **📊 Built-in Evaluation**: Comprehensive benchmarking suite (ROUGE, BLEU, BERTScore).

---

## 📈 Benchmark Results

Evaluated on the **BBC News Summary** dataset (10 samples, target length 55-120 words).

| Model Variant | ROUGE-1 | ROUGE-2 | BLEU | BERTScore (F1) |
| :--- | :---: | :---: | :---: | :---: |
| Simple Baseline (TF-IDF) | 0.3340 | 0.2227 | 0.1704 | 0.6300 |
| BART (Standard) | 0.3669 | 0.2157 | 0.0698 | 0.6631 |
| BART + RAG | 0.4383 | 0.3428 | 0.1523 | 0.7012 |
| BART + Hybrid (TextRank) | 0.5141 | 0.4180 | 0.1489 | 0.7440 |
| BART + LoRA | 0.5794 | 0.4630 | **0.3420** | 0.7485 |
| **👑 BART + Hybrid + LoRA** | **0.6355** | **0.5793** | 0.3272 | **0.7926** |

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vytora04/SmartNewsSumm.git
   cd SmartNewsSumm
   ```

2. **Initialize Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Start the Backend API (FastAPI)
```bash
uvicorn backend.app:app --reload --port 8000
```
- **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Start the Frontend UI (Streamlit)
```bash
streamlit run frontend/streamlit_app.py
```
- **UI**: [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Advanced Configuration

- **LoRA Support**: Set the path to your adapter in the environment variable `LORA_ADAPTER_PATH` or select "Enable LoRA" in the sidebar.
- **RAG Settings**: Adjust `Top K` and `Window Size` to control how much background context is fetched.
- **Repetition Control**: Use the `No Repeat N-Gram` slider (Default: 3) to tune the balance between entity retention and phrase variety.

---

## 📂 Project Structure

- `backend/`: FastAPI server and core logic (`summarizer.py`, `rag.py`, `extractive.py`).
- `frontend/`: Streamlit interactive dashboard.
- `scripts/`: Training and evaluation pipelines.
- `results/`: Cached benchmark scores and model adapters.

---

## 📝 License
MIT License. Created as part of the NLP Group 5 Project.
