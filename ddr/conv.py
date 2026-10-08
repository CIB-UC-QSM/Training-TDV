"""Operadores lineales con adjunta exacta. Port de `../tdv/ddr/conv.py`.

`backward` es la **adjunta** del operador lineal, no el gradiente de autograd: TDV es una energía
y su gradiente se escribe a mano recorriendo el grafo al revés. La identidad <Kx,y> = <x,K^T y>
se verifica en tests/test_conv.py.

Tres cosas no son intercambiables por sus equivalentes de PyTorch:
  - el padding simétrico con su transpuesta exacta (el `reflect` de PyTorch no equivale);
  - el antialias binomial [1,4,6,4,1]/16 aplicado al kernel aprendido antes del stride-2;
  - los operadores de bajada y subida son independientes, no comparten pesos.
"""


from __future__ import annotations

import numpy as np
import torch

__all__ = ["Conv2d", "ConvScale2d", "ConvScaleTranspose2d", "symmetric_pad2d", "symmetric_pad2d_transpose"]


def symmetric_pad2d(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Padding simétrico (borde incluido). `pad` es (izq, der, arriba, abajo)."""
    pad_l, pad_r, pad_t, pad_b = pad
    if pad_t > 0:
        x = torch.cat([x[..., :pad_t, :].flip(-2), x], dim=-2)
    if pad_b > 0:
        x = torch.cat([x, x[..., -pad_b:, :].flip(-2)], dim=-2)
    if pad_l > 0:
        x = torch.cat([x[..., :, :pad_l].flip(-1), x], dim=-1)
    if pad_r > 0:
        x = torch.cat([x, x[..., :, -pad_r:].flip(-1)], dim=-1)
    return x


def symmetric_pad2d_transpose(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Adjunta de `symmetric_pad2d`: acumula el sobrante en el borde espejado."""
    pad_l, pad_r, pad_t, pad_b = pad
    if pad_r > 0:
        right = x[..., :, -pad_r:].flip(-1)
        x = x[..., :, :-pad_r].clone()
        x[..., :, -pad_r:] += right
    if pad_l > 0:
        left = x[..., :, :pad_l].flip(-1)
        x = x[..., :, pad_l:].clone()
        x[..., :, :pad_l] += left
    if pad_b > 0:
        bottom = x[..., -pad_b:, :].flip(-2)
        x = x[..., :-pad_b, :].clone()
        x[..., -pad_b:, :] += bottom
    if pad_t > 0:
        top = x[..., :pad_t, :].flip(-2)
        x = x[..., pad_t:, :].clone()
        x[..., :pad_t, :] += top
    return x


class Conv2d(torch.nn.Module):
    """Convolución bias-free. `zero_mean`/`bound_norm` instalan `weight.proj()`, que hay que
    llamar después de cada paso del optimizador. En TDV sólo K1 los usa."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        invariant: bool = False,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        zero_mean: bool = False,
        bound_norm: bool = False,
        pad: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        if self.invariant:
            assert self.kernel_size == 3
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1, 3))
            mask = np.asarray([1, 4, 4], dtype=np.float32)[None, None, None, :]
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(out_channels, in_channels, kernel_size, kernel_size)
            )
            mask = np.ones((kernel_size, kernel_size), dtype=np.float32)[None, None, :, :]
        self.register_buffer("mask", torch.from_numpy(mask))

        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1 / (in_channels * kernel_size**2)))

        self.weight.L_init = 1e4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3)

            def l2_proj(surface: bool = False) -> None:
                if zero_mean:
                    mean = torch.sum(
                        self.weight.data * self.mask, self.weight.reduction_dim, True
                    ) / (self.in_channels * self.kernel_size**2)
                    self.weight.data.sub_(mean)
                if bound_norm:
                    norm = torch.sum(
                        self.weight.data**2 * self.mask, self.weight.reduction_dim, True
                    ).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(torch.max(norm, torch.ones_like(norm)))

            self.weight.proj = l2_proj
            self.weight.proj(True)

    def get_weight(self) -> torch.Tensor:
        if not self.invariant:
            return self.weight
        weight = torch.empty(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        weight[:, :, 1, 1] = self.weight[:, :, 0, 0]
        weight[:, :, ::2, ::2] = self.weight[:, :, 0, 2].view(self.out_channels, self.in_channels, 1, 1)
        weight[:, :, 1::2, ::2] = self.weight[:, :, 0, 1].view(self.out_channels, self.in_channels, 1, 1)
        weight[:, :, ::2, 1::2] = self.weight[:, :, 0, 1].view(self.out_channels, self.in_channels, 1, 1)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        pad = weight.shape[-1] // 2
        if self.pad and pad > 0:
            x = symmetric_pad2d(x.contiguous(), (pad, pad, pad, pad))
        return torch.nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def backward(self, x: torch.Tensor, output_shape: torch.Size | None = None) -> torch.Tensor:
        """Adjunta de `forward`. `output_shape` fija el tamaño exacto (dimensiones impares)."""
        weight = self.get_weight()

        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2] - 1) * self.stride + 1),
                output_shape[3] - ((x.shape[3] - 1) * self.stride + 1),
            )
        else:
            output_padding = 0

        x = torch.nn.functional.conv_transpose2d(
            x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )
        pad = weight.shape[-1] // 2
        if self.pad and pad > 0:
            x = symmetric_pad2d_transpose(x.contiguous(), (pad, pad, pad, pad))
        return x

    def extra_repr(self) -> str:
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is not None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class ConvScale2d(Conv2d):
    """Bajada de escala: antialias binomial seguido de stride-2. El binomial se aplica al kernel,
    no a la imagen, así que un 3x3 se vuelve 7x7 antes de convolucionar."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        invariant: bool = False,
        groups: int = 1,
        stride: int = 2,
        bias: bool = False,
        zero_mean: bool = False,
        bound_norm: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            invariant=invariant,
            stride=stride,
            dilation=1,
            groups=groups,
            bias=bias,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
        )

        if self.stride > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            self.register_buffer("blur", torch.from_numpy(np_k.reshape(1, 1, 5, 5)))

    def get_weight(self) -> torch.Tensor:
        weight = super().get_weight()
        if self.stride > 1:
            weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
            for _ in range(self.stride // 2):
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(
                self.out_channels,
                self.in_channels,
                self.kernel_size + 2 * self.stride,
                self.kernel_size + 2 * self.stride,
            )
        return weight


class ConvScaleTranspose2d(ConvScale2d):
    """Subida de escala: adjunta de su propio kernel. No comparte pesos con la bajada."""

    def forward(self, x: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:  # type: ignore[override]
        return super().backward(x, output_shape)

    def backward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x)
