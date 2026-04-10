"""CNN encoder: grayscale images -> sequence features for an attention-based decoder."""

from __future__ import annotations

import torch
from torch import nn


def _spatial_size_after_cnn(height: int, width: int, num_pool_layers: int = 4) -> tuple[int, int]:
    """Helper to compute spatial size after CNN blocks.

    Conv blocks use pad=1,k=3 (same H/W); each block ends with MaxPool2d(2).
    """
    assert height > 0
    assert width > 0

    return height >> num_pool_layers, width >> num_pool_layers  # equivalent to // (2 ** num_pool_layers)


def _xavier_init_module(m: nn.Module) -> None:
    """Xavier initialization for Conv2d and Linear layers using ReLU gain and zero bias."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class _ConvBNReLUPool(nn.Module):
    """Convolutional block with BatchNorm and ReLU activation, followed by MaxPooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class OCRHandwrittenCNNEncoder(nn.Module):
    """Image-to-sequence CNN encoder for handwritten OCR.

    Input: (N, 1, H, W) grayscale.
        (batch_size, 1, height, width)
    After convolutions, feature maps (N, 256, H', W') are turned into a sequence by
    treating height as time and flattening channel x width into each step.

    Output: (N, H', embedding_dim) for use with an attention-based decoder.
    """

    def __init__(
        self,
        image_height: int = 32,
        image_width: int = 256,
        embedding_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the CNN encoder with specified image dimensions and embedding size."""
        super().__init__()
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_height and image_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.image_height = image_height
        self.image_width = image_width
        self.embedding_dim = embedding_dim

        self.block1 = _ConvBNReLUPool(1, 32)
        self.block2 = _ConvBNReLUPool(32, 64)
        self.block3 = _ConvBNReLUPool(64, 128)
        self.block4 = _ConvBNReLUPool(128, 256)

        h_out, w_out = _spatial_size_after_cnn(image_height, image_width)
        if h_out == 0 or w_out == 0:
            raise ValueError(
                f"After pooling, feature map is empty for input ({image_height}, {image_width}). "
                "Increase image height and width (need at least 2^4=16 after each pooling axis)."
            )

        c_out = 256
        proj_in = c_out * w_out
        self.proj = nn.Linear(proj_in, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.apply(_xavier_init_module)

    @property
    def sequence_length(self) -> int:
        h_out, _ = _spatial_size_after_cnn(self.image_height, self.image_width)
        return h_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images into a sequence of features for the decoder."""
        if x.dim() != 4:
            raise ValueError(f"Expected (N, C, H, W); got shape {tuple(x.shape)}")
        if x.size(1) != 1:
            raise ValueError(f"Expected 1 input channel (grayscale); got C={x.size(1)}")

        # (N, 256, H', W')
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Sequence along height; flatten width and channels into each step.
        # (N, C, H', W') -> (N, H', C, W') -> (N, H', C * W')
        n, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(n, h, c * w)

        x = self.proj(x)
        return self.dropout(x)


def print_model_summary(
    model: OCRHandwrittenCNNEncoder,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> None:
    """Print parameter counts and output shape from a dummy forward pass."""
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    was_training = model.training
    model.eval()

    h, w = model.image_height, model.image_width
    dummy = torch.zeros(batch_size, 1, h, w, device=device)

    lines: list[str] = []
    lines.append("OCRHandwrittenCNNEncoder summary")
    lines.append("-" * 60)
    lines.append(f"Configured input size (H, W): ({h}, {w})")
    lines.append("Grayscale channels: 1")
    lines.append(f"Embedding dim: {model.embedding_dim}")
    lines.append(f"Expected output sequence length (H after pools): {model.sequence_length}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Total parameters: {total:,}")
    lines.append(f"Trainable parameters: {trainable:,}")

    with torch.no_grad():
        out = model(dummy)
    lines.append(f"Dummy input shape:  {tuple(dummy.shape)}")
    lines.append(f"Output shape:       {tuple(out.shape)}  (N, seq_len, embedding_dim)")
    lines.append("-" * 60)

    print("\n".join(lines))  # noqa: T201 - intentional CLI-style helper

    if was_training:
        model.train()


__all__ = ["OCRHandwrittenCNNEncoder", "print_model_summary"]
