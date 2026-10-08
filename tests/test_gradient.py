"""`TDV.grad` debe ser el gradiente exacto de `TDV.energy`, y no depender de estado mutable."""

import pytest
import torch

from ddr.tdv import TotalDeepVariation2D, student_t

CONFIG = {
    "in_channels": 3,
    "out_channels": 1,
    "num_features": 4,
    "num_scales": 3,
    "num_mb": 2,
    "multiplier": 1,
}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TotalDeepVariation2D(dict(CONFIG)).double()


def test_student_t_matches_analytic_derivative():
    x = torch.linspace(-4, 4, 101, dtype=torch.float64).requires_grad_(True)
    value, derivative = student_t(x)
    (autograd_derivative,) = torch.autograd.grad(value.sum(), x)
    assert torch.allclose(derivative, autograd_derivative, atol=1e-10)


def test_grad_matches_autograd(model):
    """La fuerza manual y la referencia de autograd deben coincidir."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)

    manual = model.grad(x.clone())
    reference = model.grad_autograd_reference(x.clone().requires_grad_(True))

    assert torch.allclose(manual, reference, atol=1e-9), (
        f"desviación máxima {(manual - reference).abs().max().item():.3e}"
    )


def test_grad_matches_numerical_directional_derivative(model):
    """Chequeo numérico por diferencias centradas a lo largo de una escala (test del repo original)."""
    torch.manual_seed(2)
    x = torch.rand(2, 3, 48, 48, dtype=torch.float64)

    directional = torch.sum(x * model.grad(x.clone())).item()

    eps = 1e-5
    with torch.no_grad():
        plus = model.energy((1 + eps) * x).sum().item()
        minus = model.energy((1 - eps) * x).sum().item()
    numerical = (plus - minus) / (2 * eps)

    assert directional == pytest.approx(numerical, rel=1e-5)


def test_grad_is_repeatable(model):
    """Sin estado mutable: llamar dos veces seguidas da el mismo resultado.

    Ojo con lo que este test NO prueba. El `MicroBlock` original guardaba `act_prime` como
    atributo del módulo, y acá las cachés se pasan explícitamente; pero este test pasa igual con
    la versión de estado mutable, porque cada `grad()` hace forward y backward de forma atómica y
    el estado siempre está fresco. El bug original sólo aparece si dos forwards se intercalan
    antes de un backward, cosa que este test no hace. Vale como no-regresión, no como respaldo
    del refactor.
    """
    torch.manual_seed(3)
    x = torch.randn(1, 3, 24, 24, dtype=torch.float64)
    assert torch.equal(model.grad(x.clone()), model.grad(x.clone()))


def test_energy_devuelve_densidad_por_pixel(model):
    """`energy()` NO devuelve un escalar por muestra: devuelve la densidad rho con forma
    [B,1,H,W]. `R(x)` es su suma sobre los píxeles, que es lo que consume `grad()`."""
    x = torch.randn(2, 3, 24, 24, dtype=torch.float64)
    assert model.energy(x).shape == (2, 1, 24, 24)


def test_analysis_kernel_projection(model):
    """K1 debe quedar con media cero y norma l2 <= 1 por filtro tras proyectar.

    El modelo se construye en float32 y luego se castea a double, asi que la proyeccion del
    `__init__` deja un residuo del orden de eps(float32). Se re-proyecta en doble precision para
    que el test mida la proyeccion en si, no el casteo.
    """
    model.K1.weight.proj(True)

    w = model.K1.weight.data
    means = w.sum(dim=(1, 2, 3))
    norms = w.flatten(1).norm(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-14)
    assert (norms <= 1.0 + 1e-12).all()
    # `surface=True` normaliza a la esfera, no a la bola
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)


def test_projection_bounds_an_oversized_kernel(model):
    """Un kernel inflado debe volver a la bola unidad; uno pequeno se deja quieto."""
    model.K1.weight.data.mul_(50.0)
    model.K1.weight.proj()
    assert (model.K1.weight.data.flatten(1).norm(dim=1) <= 1.0 + 1e-12).all()

    model.K1.weight.data.mul_(0.01)
    small = model.K1.weight.data.flatten(1).norm(dim=1).clone()
    model.K1.weight.proj()
    after = model.K1.weight.data.flatten(1).norm(dim=1)
    assert torch.allclose(small, after, atol=1e-12)


def test_no_bias_anywhere(model):
    """Bias-free de punta a punta. Junto con la media cero de K1 da invariancia a OFFSETS —
    sumarle una constante a la imagen no cambia la energía—, que es lo que dice el paper.

    NO da escala-equivariancia: eso se midió el 2026-09-06 y es falso (el teorema de Mohan et al.
    exige activación 1-homogénea y la Student-t no lo es; el esquema se desvía 15-28 %). Ver
    docs/mediciones/escala-equivariancia.txt."""
    assert not [name for name, _ in model.named_parameters() if "bias" in name]
    for module in model.modules():
        if hasattr(module, "bias"):
            assert module.bias is None


# --- SEGUNDA DERIVADA -------------------------------------------------------------------------
# El backward del entrenamiento no usa `grad()`: usa su DERIVADA. Autograd atraviesa la cadena
# adjunta escrita a mano, y produce `d(grad_R)/dx` y `d(grad_R)/dtheta`. Que la adjunta sea
# correcta (tests/test_conv.py) no implica que su derivada lo sea: son dos afirmaciones distintas.

def test_hessiano_por_vector_matches_autograd(model):
    """`d(grad_R)/dx · v`, manual contra la referencia. Es el jacobiano de cada paso desenrollado."""
    torch.manual_seed(4)
    base = torch.randn(2, 3, 24, 24, dtype=torch.float64)
    v = torch.randn_like(base)

    xa = base.clone().requires_grad_(True)
    (manual,) = torch.autograd.grad((model.grad(xa) * v).sum(), xa)

    xb = base.clone().requires_grad_(True)
    (reference,) = torch.autograd.grad((model.grad_autograd_reference(xb) * v).sum(), xb)

    assert torch.allclose(manual, reference, atol=1e-9), (
        f"desviación máxima {(manual - reference).abs().max().item():.3e}"
    )


def test_segunda_derivada_en_los_pesos_matches_autograd(model):
    """`d(grad_R)/dtheta · v`: es literalmente lo que Adam aplica a los 387 968 pesos."""
    torch.manual_seed(5)
    base = torch.randn(2, 3, 24, 24, dtype=torch.float64)
    v = torch.randn_like(base)
    pesos = [p for p in model.parameters() if p.requires_grad]

    xa = base.clone().requires_grad_(True)
    manual = torch.autograd.grad((model.grad(xa) * v).sum(), pesos)

    xb = base.clone().requires_grad_(True)
    reference = torch.autograd.grad((model.grad_autograd_reference(xb) * v).sum(), pesos)

    peor = max((m - r).abs().max().item() for m, r in zip(manual, reference))
    assert peor < 1e-9, f"desviación máxima {peor:.3e} en {len(pesos)} tensores"
