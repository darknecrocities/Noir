"""Autoregressive Causal Transformer (Language Model) architecture for open web learning."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from noir.models.base import NoirBaseModel


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with triangular upper-triangular masking."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, block_size: int = 64, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Key, Query, Value projections
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal triangular mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.resid_dropout(self.c_proj(y))


class TransformerBlock(nn.Module):
    """Standard pre-LayerNorm Transformer decoder block."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, block_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NoirTransformerLM(NoirBaseModel):
    """Causal Autoregressive Language Model instrumented for live open web learning."""

    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim

        # Token & Positional Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm & Language Modeling Head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        self.register_visualization_hooks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing token logits and cross-entropy loss."""
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Input sequence length {T} exceeds block size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Forward token + position embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens from prompt sequence."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
