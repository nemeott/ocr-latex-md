"""RNN decoder with attention for LaTeX sequence generation."""

from __future__ import annotations

import torch
import torch.nn.functional as func
from torch import nn


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention mechanism."""

    def __init__(self, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.

        Args:
            hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, seq_len, encoder_dim)

        Returns:
            context: (batch_size, encoder_dim)
            weights: (batch_size, seq_len)

        """
        # (batch_size, 1, hidden_dim)
        query = self.query_proj(hidden).unsqueeze(1)
        # (batch_size, seq_len, hidden_dim)
        keys = self.key_proj(encoder_outputs)

        # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(query + keys)
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        attention_scores = self.v(energy).squeeze(2)

        attention_weights = func.softmax(attention_scores, dim=1)

        # (batch_size, 1, seq_len) @ (batch_size, seq_len, encoder_dim) -> (batch_size, 1, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class LatexRNNDecoder(nn.Module):
    """RNN (GRU) decoder with attention for LaTeX token generation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)

        # RNN input is the concatenation of the current token embedding and the context vector
        self.rnn = nn.GRU(embedding_dim + encoder_dim, hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one time step.

        Args:
            input_token: (batch_size,) current token indices
            hidden: (batch_size, hidden_dim) previous hidden state (or init)
            encoder_outputs: (batch_size, seq_len, encoder_dim)

        Returns:
            logits: (batch_size, vocab_size)
            hidden: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)

        """
        # (batch_size, 1)
        input_token = input_token.unsqueeze(1)

        # (batch_size, 1, embedding_dim)
        embedded = self.dropout(self.embedding(input_token))

        # Get context vector from attention
        # context: (batch_size, encoder_dim)
        context, attn_weights = self.attention(hidden, encoder_outputs)

        # (batch_size, 1, encoder_dim)
        context_unsqueeze = context.unsqueeze(1)

        # (batch_size, 1, embedding_dim + encoder_dim)
        rnn_input = torch.cat((embedded, context_unsqueeze), dim=2)

        # output: (batch_size, 1, hidden_dim)
        # hidden_out: (1, batch_size, hidden_dim)
        output, hidden_out = self.rnn(rnn_input, hidden.unsqueeze(0))

        # (batch_size, hidden_dim)
        hidden_out = hidden_out.squeeze(0)
        output = output.squeeze(1)

        # Predict next token using RNN output, context, and current embedding
        # (batch_size, hidden_dim + encoder_dim + embedding_dim)
        fc_input = torch.cat((output, context, embedded.squeeze(1)), dim=1)
        logits = self.fc_out(fc_input)

        return logits, hidden_out, attn_weights


def print_decoder_summary(
    decoder: LatexRNNDecoder,
    batch_size: int = 2,
    seq_len: int = 16,
    device: torch.device | str | None = None,
) -> None:
    """Print parameter counts and output shape from a dummy forward pass."""
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    decoder = decoder.to(device)
    was_training = decoder.training
    decoder.eval()

    dummy_input = torch.zeros(batch_size, dtype=torch.long, device=device)
    dummy_hidden = torch.zeros(batch_size, decoder.hidden_dim, device=device)
    dummy_encoder_outputs = torch.zeros(batch_size, seq_len, decoder.encoder_dim, device=device)

    lines: list[str] = []
    lines.append("LatexRNNDecoder summary")
    lines.append("-" * 60)
    lines.append(f"Vocab size: {decoder.vocab_size}")
    lines.append(f"Embedding dim: {decoder.embedding_dim}")
    lines.append(f"Hidden dim: {decoder.hidden_dim}")
    lines.append(f"Encoder dim: {decoder.encoder_dim}")

    total = sum(p.numel() for p in decoder.parameters())
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    lines.append(f"Total parameters: {total:,}")
    lines.append(f"Trainable parameters: {trainable:,}")

    with torch.no_grad():
        logits, next_hidden, attn = decoder(dummy_input, dummy_hidden, dummy_encoder_outputs)

    lines.append(
        f"Dummy inputs: token={tuple(dummy_input.shape)}, hidden={tuple(dummy_hidden.shape)}, enc_outs={tuple(dummy_encoder_outputs.shape)}"
    )
    lines.append(f"Logits output shape: {tuple(logits.shape)}  (N, vocab_size)")
    lines.append(f"Next hidden shape:   {tuple(next_hidden.shape)}  (N, hidden_dim)")
    lines.append(f"Attention shape:     {tuple(attn.shape)}  (N, seq_len)")
    lines.append("-" * 60)

    print("\n".join(lines))

    if was_training:
        decoder.train()
