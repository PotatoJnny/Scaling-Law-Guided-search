"""Download Qwen2.5-Math-7B-Instruct to the Hugging Face cache.
Run from the repository root: python scripts/download_qwen25math7b.py
"""
from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen2.5-Math-7B-Instruct...")
path = snapshot_download(repo_id="Qwen/Qwen2.5-Math-7B-Instruct")
print(f"Downloaded to: {path}")
