import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from compression import apply_operator_fusion
from kv_cache import KVCacheAttention
from models import SimpleNet


def test_simplenet_forward():
    model = SimpleNet()
    x = torch.randn(1, 784)
    out = model(x)
    assert out.shape == (1, 10)


def test_operator_fusion_returns_script_module():
    model = SimpleNet()
    fused = apply_operator_fusion(model)
    assert hasattr(fused, "graph")


def test_kv_cache_runs():
    torch.manual_seed(0)
    attn = KVCacheAttention()
    x = torch.randn(1, 1, 32)
    out1 = attn(x, use_cache=True)
    out2 = attn(x, use_cache=True)
    assert out2.shape == (1, 1, 32)
    assert not torch.allclose(out1, out2)
