# quantize

Demonstrates strategies to compress a PyTorch model for efficient inference and showcases optimizations like operator fusion and key‑value caching.

## Environment

Install runtime and development dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --group dev
```

## Scripts

| Script | Description | Command |
| --- | --- | --- |
| `train.py` | Train a model and export several compressed variants. | `uv run python train.py` |
| `fusion.py` | Load `model.pth` and write a TorchScript‑fused version. | `uv run python fusion.py` |
| `kv_cache.py` | Demonstrate key‑value cache usage in attention. | `uv run python kv_cache.py` |

## Compression and inference techniques

| Technique | How it works | Notes |
| --- | --- | --- |
| FP16 weights | Cast parameters to 16‑bit floats before saving. | Halves storage requirements. |
| Dynamic quantization | Convert `Linear` layers to INT8 at inference. | Limited layer support, small accuracy hit. |
| Structured pruning | Remove channels using `ln` structured pruning. | May need fine‑tuning to recover accuracy. |
| Low‑rank factorization | Replace linear layers with low‑rank pairs. | Choose rank to balance size vs. error. |
| Knowledge distillation | Train smaller student to mimic teacher. | Requires extra training time. |
| Operator fusion | TorchScript combines adjacent ops. | Model must be in evaluation mode. |
| KV‑cache | Reuse attention keys/values for decoding. | Trades memory for speed on long sequences. |

## Development

Run unit tests with:

```bash
uv run pytest
```

Lint and format staged files using pre‑commit:

```bash
uv run pre-commit run --files <files>
```

These checks also run automatically in GitHub Actions.

