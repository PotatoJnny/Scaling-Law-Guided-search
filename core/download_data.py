from datasets import load_dataset

dataset = load_dataset("math-ai/aime25", split='test')
dataset.save_to_disk("./local_aime_2025_data")