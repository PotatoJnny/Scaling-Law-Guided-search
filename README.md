# Scaling-Law Guided (SLG) Search

This repository implements the **Scaling-Law Guided (SLG) Search** algorithm. 
Utilizing the scaling law predicted by tail estimation, we dynamically allocate test-time compute to maximize the probability of finding high-reward solutions.


## 📂 File Structure

```text
SLG/
├── one_layer.py            # Main entry point for the search execution
├── Algorithm/
│   ├── algorithms.py       # Core logic for the SLG_Search class
│   └── node.py             # Implementation of the node
└── core/
    ├── download_data.py    # Utilities for downloading datasets
    ├── tools.py            # Helper functions (prompting, data cleaning)
    ├── data_structures.py  # Definitions for State and Action objects
    ├── model_config.py     # Configuration management for LM, RM, and Search parameters
    └── model_wrapper.py    # Wrappers for Large Language Models (LLM) and Reward Models (RM)
```

## ⚙️ Setup

Install the required dependencies:

```bash
pip install torch transformers datasets scipy pandas numpy bitsandbytes accelerate
```


## 🚀 Usage

The main entry point is `one_layer.py`. This script executes the SLG search on a specified dataset.

### 1. Basic Run (Auto-Configuration)

Executes the search using default models. The script will automatically calculate the optimal search width ($K$) and estimation samples ($m$) based on the total budget ($N$).

```bash
python one_layer.py --dataset "math-ai/aime25" --N 500
```

### 2. Advanced Run (Manual Configuration)

Allows for manual control over the search parameters and model selection:

```bash
python one_layer.py \
    --dataset "math-ai/aime25" \
    --lm_name "meta-llama/Llama-3.2-1B-Instruct" \
    --rm_name "Skywork/Skywork-Reward-V2-Llama-3.1-8B" \
    --N 1000 \
    --K 8 \
    --m 80
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `math-ai/aime25` | HuggingFace dataset path (e.g., `math-ai/aime25`). |
| `--N` | `int` | `1000` | **Total Budget**. The maximum number of rollouts allowed per problem. |
| `--K` | `int` | *Auto* | **Search Width**. The number of intermediate states to explore. |
| `--m` | `int` | *Auto* | **Estimation Samples**. The number of rollouts per state used to estimate potential value. |
| `--lm_name` | `str` | `Llama-3.2-1B` | The Policy/Language Model to use. |
| `--rm_name` | `str` | `Skywork-Reward` | The Reward Model to use. |


## 🧠 Algorithm Logic

**Input:** Prompt $x$, Total Budget $N$, Search Width $K$, Estimation Samples $m$.

**Output:** The response $y$ with the highest reward.

1.  **Expand**: Generate $K$ intermediate states $\{s_1, \dots, s_K\}$ from prompt $x$.
2.  **Estimate**: For each state $s_i$:
    * Generate $m$ responses $\{y_{i,1}, \dots, y_{i,m}\}$ and observe their rewards.
    * Estimate the potential value $\hat{V}_N(s_i)$ using the scaling law estimator.
3.  **Select**: Identify the optimal intermediate state:
    $$\hat{I} = \arg\max_{i \in [K]} \hat{V}_N(s_i)$$
4.  **Exploit**: Allocate the remaining budget ($N - K \times m$) to state $s_{\hat{I}}$ to generate additional responses.
5.  **Return**: The response $y$ with the highest observed reward among all generated responses.

## 📊 Output

Results are automatically saved in the `Results/` directory. The file path includes the experiment configuration to ensure reproducibility:

```text
Results/{dataset_name}/SLG_LM-{lm}_RM-{rm}_K-{K}_m-{m}_N-{N}/results.json
```
