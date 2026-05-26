# RLVR Datasets and Where to Find Them: Tracing Data Lineage for Better Training Data

Official implementation of the paper **"RLVR Datasets and Where to Find Them: Tracing Data Lineage for Better Training Data"**.

Our training framework is built upon the [verl](https://github.com/volcengine/verl) library. We implement the evaluation pipeline as a recipe extension under `verl/recipe/eval`, while training is launched directly through shell scripts.

---

### Installation

We follow the official installation guide of [verl](https://github.com/volcengine/verl) to set up the environment from scratch.

```bash
conda create -n verl python==3.10
conda activate verl

cd verl
pip install --no-deps -e .

pip install datasets==4.0.0
pip install math_verify
````

We also provide a `requirements.txt` file corresponding to our local environment (CUDA 12.4).
Please note that this file is provided for reference only, and we do not recommend installing dependencies directly from it.

---

### 1. Data Canonicalization

Please refer to the Python scripts under `./ATLAS/data_canonicalize` to generate the required data files.

Example command for preparing training data (using DAPO as an example):

```bash
python ./ATLAS/data_canonicalize/train_math/DAPO-Math-17k.py
```

---

### 2. Data Lineage Tracing

The complete lineage tracing pipeline is implemented in several stages under `./ATLAS/`.

Please refer to the corresponding scripts for each stage.

---

### 3. GRPO Training

Before training, configure the paths for `HOME`, `STORAGE`, and your `WANDB_API_KEY` in the training scripts.

Then launch training with:

```bash
bash ./verl/scripts/train/GRPO_qwen3-1.7b-base.sh
bash ./verl/scripts/train/GRPO_qwen3-8b-base.sh
```

Checkpoints will be saved to:

```bash
$HOME/checkpoints/ATLAS/$exp_name
```

---

### 4. Evaluation

Run evaluation with:

```bash
bash ./verl/scripts/eval/all-in-one_for_best_ckpt.sh
```

---

### 5. Compute Quality Scores

```bash
python ./Benchmark_Scoring/analyze_features.py
python ./Benchmark_Scoring/compute_scores.py
```

---

### Acknowledgement

Our codebase is built upon the [verl](https://github.com/volcengine/verl) framework for RL training, and the evaluation pipeline is inspired by the [LIMO](https://github.com/GAIR-NLP/LIMO) project.