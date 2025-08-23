import torch
from torch import nn


class KVCacheAttention(nn.Module):
    """Minimal attention layer with key-value caching."""

    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache: bool = False):  # noqa: D401 - basic forward
        """Run attention, optionally updating the KV cache."""
        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, x], dim=1)
            v = torch.cat([self.cache_v, x], dim=1)
        else:
            k = v = x
        out, _ = self.attn(x, k, v, need_weights=False)
        if use_cache:
            self.cache_k = k
            self.cache_v = v
        return out


def main() -> None:
    torch.manual_seed(0)
    attn = KVCacheAttention()
    x = torch.randn(1, 1, 32)
    for step in range(3):
        x = attn(x, use_cache=True)
        print(f"KV cache step {step} output norm: {x.norm():.2f}")


if __name__ == "__main__":
    main()
