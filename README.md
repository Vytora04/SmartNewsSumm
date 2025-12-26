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
