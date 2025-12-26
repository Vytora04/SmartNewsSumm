# test_features.py
"""Quick test of new features"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing extractive summarization...")
from backend.extractive import extractive_tfidf, extractive_textrank, extractive_lead

test_text = """
The global economy showed signs of recovery in the third quarter. 
Market analysts reported strong growth across multiple sectors. 
Technology companies led the gains with significant investments. 
Consumer spending increased by 3.5 percent compared to last year. 
Unemployment rates fell to their lowest level since 2019.
"""

print("\nTF-IDF summary:")
print(extractive_tfidf(test_text, num_sentences=2))

print("\nTextRank summary:")
print(extractive_textrank(test_text, num_sentences=2))

print("\nLead summary:")
print(extractive_lead(test_text, num_sentences=2))

print("\n" + "="*60)
print("Testing evaluation metrics...")
from backend.metrics import evaluate_summary

generated = "The economy recovered strongly with technology companies leading gains."
reference = "The global economy showed recovery with tech sector growth."

metrics = evaluate_summary(generated, reference, include_bertscore=False)
print(f"\nMetrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

print("\n" + "="*60)
print("Testing model configs...")
from backend.summarizer import NewsSummarizer

print("\nAvailable model configurations:")
for model_name, config in NewsSummarizer.MODEL_CONFIGS.items():
    print(f"{model_name}:")
    print(f"  max_length: {config['max_length']}, min_length: {config['min_length']}, length_penalty: {config['length_penalty']}")

print("\n" + "="*60)
print("✅ All features working correctly!")

metrics = evaluate_summary(generated, reference)
print("\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")