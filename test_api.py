"""Test new API features: model selection and extractive methods"""
import requests
import time

API_URL = "http://127.0.0.1:8000/summarize"

test_article = """
Global technology companies announced major investments in renewable energy this week. 
Microsoft committed $10 billion to solar infrastructure across North America. 
Google unveiled plans for wind-powered data centers in Europe. 
Amazon revealed a new carbon-neutral delivery fleet rollout. 
Industry analysts predict these moves will accelerate the clean energy transition.
"""

print("Testing new API features...")
print("="*80)

# Wait for backend to start
print("\nWaiting for backend to start...")
time.sleep(5)

# Test 1: Abstractive with BART
print("\n1. Testing BART Large...")
try:
    response = requests.post(API_URL, json={
        "text": test_article,
        "model_name": "facebook/bart-large-cnn",
        "input_is_html": False,
        "run_qa": False  # Skip QA for speed
    }, timeout=60)
    if response.status_code == 200:
        summary = response.json()["summary"]
        print(f"BART Summary: {summary}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Extractive TF-IDF
print("\n2. Testing Extractive TF-IDF...")
try:
    response = requests.post(API_URL, json={
        "text": test_article,
        "extractive_method": "tfidf",
        "input_is_html": False
    }, timeout=30)
    if response.status_code == 200:
        summary = response.json()["summary"]
        print(f"✅ TF-IDF Summary: {summary}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: Extractive TextRank
print("\n3. Testing Extractive TextRank...")
try:
    response = requests.post(API_URL, json={
        "text": test_article,
        "extractive_method": "textrank",
        "input_is_html": False
    }, timeout=30)
    if response.status_code == 200:
        summary = response.json()["summary"]
        print(f"TextRank Summary: {summary}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")

# Test 4: Extractive Lead
print("\n4. Testing Extractive Lead...")
try:
    response = requests.post(API_URL, json={
        "text": test_article,
        "extractive_method": "lead",
        "input_is_html": False
    }, timeout=30)
    if response.status_code == 200:
        summary = response.json()["summary"]
        print(f"Lead Summary: {summary}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")

print("\n" + "="*80)
print("All API tests complete!")
