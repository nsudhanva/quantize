# quantize

Demonstrates strategies to compress a PyTorch model for efficient inference.

## Environment

This project uses [uv](https://github.com/astral-sh/uv) for managing
dependencies and virtual environments. To set up the project with the
dependencies listed in `pyproject.toml`:

```bash
uv sync
```

Run the example training and quantization script with:

```bash
uv run python model.py
```

## Compression techniques

The script trains a small model and then applies several popular techniques.
Each method saves weights to disk and prints their sizes so you can compare the
space savings and consider accuracy or hardware trade‑offs.

### Dynamic quantization
* **How it works:** Converts linear layers to 8‑bit integers at inference time.
* **Advantages:** Post‑training and fast to apply. Reduces weight size by 4×.
* **Trade‑offs:** Limited to certain layer types and may slightly reduce
  accuracy, especially on activations with large dynamic ranges.

### Structured pruning
* **How it works:** Removes entire channels (50% in this example) using
  structured `ln` pruning.
* **Advantages:** Produces sparse models that may run faster on hardware that
  exploits sparsity.
* **Trade‑offs:** Pruned models often require fine‑tuning to recover accuracy,
  and unstructured hardware may not see speedups.

### Knowledge distillation
* **How it works:** Trains a smaller student network to mimic the teacher's
  outputs and then applies dynamic quantization.
* **Advantages:** Shrinks the architecture itself while retaining much of the
  teacher's behavior.
* **Trade‑offs:** Requires additional training time for the student and depends
  on a well‑trained teacher.

### FP16 weights
* **How it works:** Casts model weights to 16‑bit floating point before saving.
* **Advantages:** Halves storage and memory bandwidth relative to FP32 and is
  widely supported on GPUs.
* **Trade‑offs:** Some CPUs lack fast FP16 support and the lower precision can
  harm accuracy for models sensitive to numerical range.

### Low‑rank factorization
* **How it works:** Approximates each linear layer with two smaller matrices
  derived from a truncated SVD decomposition.
* **Advantages:** Reduces parameter count and may expose additional sparsity.
* **Trade‑offs:** Adds extra layers and multiplications, and the rank choice
  balances size reduction against approximation error.
