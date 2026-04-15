"""Training script for OCR CNN Encoder and LaTeX RNN Decoder."""

import torch
from latex_rnn_decoder import LatexRNNDecoder
from ocr_cnn_encoder import OCRHandwrittenCNNEncoder
from torch import nn, optim


def train_step(
    images: torch.Tensor,
    targets: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """Perform a single training step (one batch).

    Args:
        images: (batch_size, 1, H, W) grayscale images.
        targets: (batch_size, max_seq_len) target token indices. targets[:, 0] should be BOS.
        encoder: OCRHandwrittenCNNEncoder.
        decoder: LatexRNNDecoder.
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g., CrossEntropyLoss with ignore_index).
        teacher_forcing_ratio: Probability of using teacher forcing.

    Returns:
        Average loss for the batch.
    """
    batch_size = images.size(0)
    max_seq_len = targets.size(1)
    device = images.device

    optimizer.zero_grad()

    # 1. Pass image through encoder
    # encoder_outputs shape: (batch_size, enc_seq_len, encoder_dim)
    encoder_outputs = encoder(images)

    # 2. Initialize decoder hidden state
    # hidden shape: (batch_size, hidden_dim)
    hidden = torch.zeros(batch_size, decoder.hidden_dim, device=device)

    # 3. Initialize decoder input with BOS token (assumed to be at targets[:, 0])
    input_token = targets[:, 0]

    loss = 0.0
    use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

    # 4. Step through the sequence
    for t in range(1, max_seq_len):
        logits, hidden, _ = decoder(input_token, hidden, encoder_outputs)

        # targets[:, t] is the ground truth next token
        step_target = targets[:, t]
        loss += criterion(logits, step_target)

        if use_teacher_forcing:
            # Teacher forcing: feed the ground truth token as the next input
            input_token = step_target
        else:
            # Without teacher forcing: use the model's own highest probability prediction
            _, topi = logits.topk(1)
            input_token = topi.squeeze(1).detach()

    # 5. Backpropagation
    loss.backward()

    # Optional: gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item() / (max_seq_len - 1)


def main() -> None:
    """Run a dummy training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = 500
    embedding_dim = 256
    hidden_dim = 512
    encoder_dim = 512
    pad_id = 0
    bos_id = 1
    eos_id = 2

    # Initialize models
    encoder = OCRHandwrittenCNNEncoder(
        image_height=32,
        image_width=256,
        embedding_dim=encoder_dim,
        dropout=0.1,
    ).to(device)

    decoder = LatexRNNDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_dim=encoder_dim,
        dropout=0.1,
    ).to(device)

    # Setup Optimizer and Loss
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Create dummy data
    batch_size = 8
    max_seq_len = 32
    images = torch.randn(batch_size, 1, 32, 256).to(device)

    # Target tokens: BOS followed by random tokens, maybe some padding at the end
    targets = torch.randint(3, vocab_size, (batch_size, max_seq_len)).to(device)
    targets[:, 0] = bos_id  # Set BOS token
    targets[:, -1] = eos_id  # Set EOS token (for completeness, though might not be hit if seq ends early)

    encoder.train()
    decoder.train()

    epochs = 5
    for epoch in range(1, epochs + 1):
        # In a real script, you would loop over a DataLoader here.
        avg_loss = train_step(
            images=images,
            targets=targets,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            criterion=criterion,
            teacher_forcing_ratio=0.5,
        )
        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

    print("Dummy training completed successfully.")


if __name__ == "__main__":
    main()
