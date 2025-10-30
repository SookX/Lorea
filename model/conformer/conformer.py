from typing import Optional, Tuple

import torch
import torch.nn as nn


__all__ = ["Conformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask



import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        squeezed = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv1d(in_channels, squeezed, 1)
        self.fc2 = nn.Conv1d(squeezed, in_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class ResidualConvBlock(nn.Module):
    """Residual 1D convolutional block with inner skip and SE module."""
    def __init__(
        self,
        input_dim,
        num_channels,
        depthwise_kernel_size,
        dropout=0.1,
        bias=True,
        use_group_norm=True,
    ):
        super().__init__()

        # Pointwise conv + GLU
        self.pw_conv_in = nn.Conv1d(input_dim, 2 * num_channels, 1, bias=bias)
        self.glu = nn.GLU(dim=1)

        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            num_channels,
            num_channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=num_channels,
            bias=bias,
        )

        # Norm + Activation
        self.norm = (
            nn.GroupNorm(1, num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels)
        )
        self.act = nn.SiLU()

        # Squeeze-Excitation
        self.se = SqueezeExcitation(num_channels)

        # Output conv
        self.pw_conv_out = nn.Conv1d(num_channels, input_dim, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Outer residual scale
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # Input: (B, C, T)
        x_pw = self.glu(self.pw_conv_in(x))
        x_dw = self.dw_conv(x_pw)
        x_dw = x_dw + x_pw  # inner residual
        x_dw = self.norm(x_dw)
        x_dw = self.act(x_dw)
        x_dw = self.se(x_dw)
        out = self.pw_conv_out(x_dw)
        out = self.dropout(out)
        return x + self.scale * out  # outer residual


class _ConvolutionModule(nn.Module):
    """Conformer convolution module with ResidualConvBlock."""
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ):
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")

        # Pre-LayerNorm as in Conformer
        self.layer_norm = nn.LayerNorm(input_dim)

        # Replace the sequential with our improved residual block
        self.conv_block = ResidualConvBlock(
            input_dim=input_dim,
            num_channels=num_channels,
            depthwise_kernel_size=depthwise_kernel_size,
            dropout=dropout,
            bias=bias,
            use_group_norm=use_group_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (B, T, D)
        Returns:
            torch.Tensor: shape (B, T, D)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)      # (B, D, T)
        x = self.conv_block(x)     # apply residual conv block
        return x.transpose(1, 2)   # back to (B, T, D)



class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths