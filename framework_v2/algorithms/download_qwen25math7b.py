"""Download Qwen2.5-Math-7B-Instruct to shared HF cache.
Run from login node: python scripts/download_qwen25math7b.py
"""
import os
os.environ["HF_HOME"] = "/project/6101845/shared/huggingface_cache"

from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen2.5-Math-7B-Instruct...")
path = snapshot_download(repo_id="Qwen/Qwen2.5-Math-7B-Instruct")
print(f"Downloaded to: {path}")
