"""La identidad adjunta <Kx, y> == <x, K^T y> para cada operador lineal.

Es el test que sostiene todo lo demás: si un `backward` no es la adjunta exacta de su `forward`,
la "fuerza" de TDV deja de ser el gradiente de una energía y el andamiaje teórico del paper
(estado adjunto, análisis de estabilidad) no aplica.
"""

import pytest
import torch

from ddr.conv import Conv2d, ConvScale2d, ConvScaleTranspose2d, symmetric_pad2d, symmetric_pad2d_transpose


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sum(a * b).item()


@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (2, 3, 31, 29)])
def test_symmetric_pad_adjoint(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64)
    pad = (1, 1, 1, 1)
    y = torch.randn_like(symmetric_pad2d(x, pad))

    lhs = _dot(symmetric_pad2d(x, pad), y)
    rhs = _dot(x, symmetric_pad2d_transpose(y, pad))
    assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_conv2d_adjoint(kernel_size):
    torch.manual_seed(0)
    op = Conv2d(3, 8, kernel_size=kernel_size, bias=False).double()
    x = torch.randn(2, 3, 24, 24, dtype=torch.float64)
    y = torch.randn_like(op.forward(x))

    lhs = _dot(op.forward(x), y)
    rhs = _dot(x, op.backward(y, x.shape))
    assert lhs == pytest.approx(rhs, rel=1e-9)


@pytest.mark.parametrize("size", [24, 25])
def test_conv_scale_adjoint(size):
    """Incluye tamaño impar: el `output_padding` debe recuperar la forma exacta."""
    torch.manual_seed(0)
    op = ConvScale2d(4, 8, kernel_size=3, bias=False).double()
    x = torch.randn(2, 4, size, size, dtype=torch.float64)
    y = torch.randn_like(op.forward(x))

    lhs = _dot(op.forward(x), y)
    rhs = _dot(x, op.backward(y, x.shape))
    assert lhs == pytest.approx(rhs, rel=1e-9)


@pytest.mark.parametrize("size", [24, 25])
def test_conv_scale_transpose_adjoint(size):
    torch.manual_seed(0)
    op = ConvScaleTranspose2d(4, 8, kernel_size=3, bias=False).double()
    coarse = torch.randn(2, 8, (size + 1) // 2, (size + 1) // 2, dtype=torch.float64)
    fine_shape = torch.Size((2, 4, size, size))

    up = op.forward(coarse, fine_shape)
    assert up.shape == fine_shape

    y = torch.randn_like(up)
    lhs = _dot(up, y)
    rhs = _dot(coarse, op.backward(y))
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_binomial_blur_es_el_binomial():
    """El antialias tiene que ser el binomial [1,4,6,4,1] normalizado, no un kernel cualquiera.

    Comprobar sólo que sume 1 no alcanza: una caja `ones(5,5)/25` pasaría esa prueba y también
    los ocho tests de adjunta de este archivo, porque `<Kx,y> = <x,K'y>` no mira los valores.
    Es la restricción 7 del README, listada como no negociable."""
    op = ConvScale2d(1, 1, kernel_size=3, bias=False)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    esperado = torch.outer(k, k) / 256.0
    assert op.blur.shape == (1, 1, 5, 5)
    assert torch.allclose(op.blur[0, 0].double(), esperado.double(), atol=1e-7)
    assert op.blur.sum().item() == pytest.approx(1.0)
    # kernel 3x3 desenfocado con el binomial 5x5 -> 7x7
    assert op.get_weight().shape[-1] == 3 + 2 * op.stride
