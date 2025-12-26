# SmartNewsSumm

An intelligent news summarization system with hybrid extract-then-abstract architecture for high-quality summaries.

## Features

- **Hybrid Mode**: Combines extractive and abstractive summarization for superior quality and speed
- **RAG Integration**: Retrieval-augmented generation for context-aware summaries
- **Claim Generation**: Automatic fact extraction from articles
- **Multiple Models**: Support for DistilBART, PEGASUS, and BART variants
- **Extractive Methods**: TextRank, LexRank, and LSA-based sentence selection
- **Auto Configuration**: Smart parameter tuning based on article length
- **Length Control**: Configurable summary length with percentage or word count
- **Interactive UI**: Streamlit-based web interface

## Performance

- **Hybrid Mode** (default): Achieves substantially better ROUGE scores with faster processing
- Filters articles through TextRank extractive layer before abstractive summarization
- Processes summaries about a third faster than pure abstractive methods

## Installation

1. Activate virtual environment:
```bash
venv/Scripts/Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Backend API
```bash
uvicorn backend.app:app --reload --port 8000
```

### Run Streamlit Frontend
```bash
streamlit run frontend/streamlit_app.py
```

### Access the Application
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## Configuration

Hybrid mode is enabled by default in the UI. You can adjust:
- **Hybrid Mode**: Toggle extract-then-abstract processing
- **Filter Strength**: Control how much content to retain (20-50%)
- **Summary Length**: Set target length as percentage or word count
- **Extractive Method**: Choose between TextRank, LexRank, or LSA
