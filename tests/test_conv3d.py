"""La identidad adjunta <Kx, y> == <x, K^T y> para cada operador lineal 3D. Espejo de test_conv.py.

Si un `backward` no es la adjunta exacta de su `forward`, la "fuerza" de TDV deja de ser el
gradiente de una energía. Las formas mezclan ejes pares e impares: el `output_padding` de la
subida se calcula eje por eje, y un error en uno solo no lo vería un cubo.
"""

import numpy as np
import pytest
import torch

from ddr.conv3d import (
    Conv3d,
    ConvScale3d,
    ConvScaleTranspose3d,
    symmetric_pad3d,
    symmetric_pad3d_transpose,
)

# (izq, der, arriba, abajo, adelante, atrás). El 3 es el que usan los kernels 7x7x7 de la bajada.
PADS = [(1, 1, 1, 1, 1, 1), (3, 3, 3, 3, 3, 3), (1, 2, 0, 3, 2, 1)]
SHAPES = [(2, 3, 12, 10, 8), (2, 3, 11, 9, 7), (1, 2, 12, 9, 10)]


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sum(a * b).item()


@pytest.mark.parametrize("pad", PADS)
def test_symmetric_pad3d_es_el_simetrico(pad):
    """La adjunta no mira los valores: un `reflect` (sin repetir el borde) pasaría el test de
    abajo. Se compara contra `np.pad(mode="symmetric")`, que repite el borde."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 6, 5, dtype=torch.float64)
    pad_l, pad_r, pad_t, pad_b, pad_front, pad_back = pad
    ancho = ((0, 0), (0, 0), (pad_front, pad_back), (pad_t, pad_b), (pad_l, pad_r))
    esperado = np.pad(x.numpy(), ancho, mode="symmetric")
    assert torch.equal(symmetric_pad3d(x, pad), torch.from_numpy(esperado))


@pytest.mark.parametrize("pad", PADS)
@pytest.mark.parametrize("shape", SHAPES)
def test_symmetric_pad3d_adjoint(shape, pad):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(symmetric_pad3d(x, pad))

    lhs = _dot(symmetric_pad3d(x, pad), y)
    rhs = _dot(x, symmetric_pad3d_transpose(y, pad))
    assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kernel_size", [1, 3])
def test_conv3d_adjoint(kernel_size, shape):
    torch.manual_seed(0)
    op = Conv3d(shape[1], 8, kernel_size=kernel_size, bias=False).double()
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(op.forward(x))

    lhs = _dot(op.forward(x), y)
    rhs = _dot(x, op.backward(y, x.shape))
    assert lhs == pytest.approx(rhs, rel=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_scale3d_adjoint(shape):
    """Incluye ejes impares: el `output_padding` debe recuperar la forma exacta."""
    torch.manual_seed(0)
    op = ConvScale3d(shape[1], 4, kernel_size=3, bias=False).double()
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(op.forward(x))

    lhs = _dot(op.forward(x), y)
    rhs = _dot(x, op.backward(y, x.shape))
    assert lhs == pytest.approx(rhs, rel=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_scale_transpose3d_adjoint(shape):
    torch.manual_seed(0)
    op = ConvScaleTranspose3d(shape[1], 4, kernel_size=3, bias=False).double()
    coarse = torch.randn(shape[0], 4, *[(n + 1) // 2 for n in shape[2:]], dtype=torch.float64)
    fine_shape = torch.Size(shape)

    up = op.forward(coarse, fine_shape)
    assert up.shape == fine_shape

    y = torch.randn_like(up)
    lhs = _dot(up, y)
    rhs = _dot(coarse, op.backward(y))
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_binomial_blur3d_es_el_binomial():
    """El antialias tiene que ser el binomial [1,4,6,4,1] en los tres ejes, normalizado: una caja
    `ones(5,5,5)/125` también suma 1 y pasa todos los tests de adjunta de este archivo."""
    op = ConvScale3d(1, 1, kernel_size=3, bias=False)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64)
    esperado = torch.einsum("i,j,k->ijk", k, k, k) / 16.0**3
    assert op.blur.shape == (1, 1, 5, 5, 5)
    assert torch.equal(op.blur[0, 0].double(), esperado)
    assert op.blur.sum().item() == pytest.approx(1.0)
    # kernel 3x3x3 desenfocado con el binomial 5x5x5 -> 7x7x7
    assert op.get_weight().shape[-3:] == (3 + 2 * op.stride,) * 3


@pytest.mark.parametrize("in_channels", [1, 3])
def test_proyeccion_k1(in_channels):
    """La proyección de K1: media cero sobre los `in_channels * 27` pesos de cada filtro y norma
    l2 <= 1. Con un divisor mal (el `k**2` del 2D) la media no quedaría en cero.

    Se construye en float32 y se castea a double, así que se re-proyecta en doble precisión para
    medir la proyección y no el casteo."""
    torch.manual_seed(0)
    op = Conv3d(in_channels, 32, 3, zero_mean=True, bound_norm=True).double()
    assert op.weight.reduction_dim == (1, 2, 3, 4)
    op.weight.proj(True)

    w = op.weight.data
    means = w.sum(dim=(1, 2, 3, 4))
    norms = w.flatten(1).norm(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-14)
    # `surface=True` normaliza a la esfera, no a la bola
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)

    # un kernel inflado vuelve a la bola unidad; uno chico se deja quieto
    op.weight.data.mul_(50.0)
    op.weight.proj()
    assert (op.weight.data.flatten(1).norm(dim=1) <= 1.0 + 1e-12).all()

    op.weight.data.mul_(0.01)
    small = op.weight.data.flatten(1).norm(dim=1).clone()
    op.weight.proj()
    assert torch.allclose(small, op.weight.data.flatten(1).norm(dim=1), atol=1e-12)


def test_solo_las_restricciones_instalan_proj():
    """Como en 2D: `proj` existe sólo con `zero_mean` o `bound_norm`, y `project_()` de la VNet
    lo busca con `hasattr`. Un `proj` de más proyectaría pesos que el paper deja libres."""
    assert not hasattr(Conv3d(4, 4).weight, "proj")
    assert not hasattr(ConvScale3d(4, 4).weight, "proj")
    assert Conv3d(4, 4).weight.L_init == 1e4
