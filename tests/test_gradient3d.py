"""`TDV3D.grad` debe ser el gradiente exacto de `TDV3D.energy`. Espejo de test_gradient.py.

Volumen de 13x14x15: los tres ejes distintos y dos impares, para que una transpuesta que confunda
ejes o pierda un vóxel en la subida no pase. Con `num_scales=3` baja a 7x7x8 y a 4x4x4.
"""

import copy
from pathlib import Path

import pytest
import torch

import train
from ddr.model import VNet2D, VNet3D
from ddr.tdv import TotalDeepVariation2D
from ddr.tdv3d import TotalDeepVariation3D

CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "num_features": 4,
    "num_scales": 3,
    "num_mb": 2,
    "multiplier": 1,
}
VOLUMEN = (2, 1, 13, 14, 15)

# TDV33 en gris, derivado: K1 1·32·27 = 864; cada macro-bloque, 5 micro-bloques de 2·32·32·27 =
# 55 296 y 4 cambios de escala de 32·32·27 = 27 648, 387 072; K_N 32·1 = 32. 864 + 3·387 072 + 32.
# Es el triple del 2D gris (387 392) salvo K_N, que es 1x1x1 en los dos.
PARAMETROS_TDV33_GRIS = 1_162_112

CHECKPOINT_GRIS = Path("checkpoints/Oficiales/tdv3-3-25-f32-gray.pth")


def vnet_config(r_config: dict, steps: int) -> dict:
    """La config de `train.vnet_config`, con T_INIT y LAMBDA_INIT de la receta, pero con otro
    regularizador y otro S."""
    config = train.vnet_config(train.T_INIT, channels=r_config["in_channels"])
    config["S"] = steps
    config["R"]["config"] = dict(r_config)
    return config


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TotalDeepVariation3D(dict(CONFIG)).double()


def test_grad_matches_autograd(model):
    """La fuerza manual y la referencia de autograd deben coincidir."""
    torch.manual_seed(1)
    x = torch.randn(*VOLUMEN, dtype=torch.float64)

    manual = model.grad(x.clone())
    reference = model.grad_autograd_reference(x.clone().requires_grad_(True))

    assert torch.allclose(manual, reference, atol=1e-9), (
        f"desviación máxima {(manual - reference).abs().max().item():.3e}"
    )


def test_grad_matches_numerical_directional_derivative(model):
    """Chequeo numérico por diferencias centradas a lo largo de una escala."""
    torch.manual_seed(2)
    x = torch.rand(*VOLUMEN, dtype=torch.float64)

    directional = torch.sum(x * model.grad(x.clone())).item()

    eps = 1e-5
    with torch.no_grad():
        plus = model.energy((1 + eps) * x).sum().item()
        minus = model.energy((1 - eps) * x).sum().item()
    numerical = (plus - minus) / (2 * eps)

    assert directional == pytest.approx(numerical, rel=1e-5)


def test_grad_is_repeatable(model):
    """Sin estado mutable: llamar dos veces seguidas da el mismo resultado. Vale como
    no-regresión, no como prueba del refactor de las cachés (ver test_gradient.py)."""
    torch.manual_seed(3)
    x = torch.randn(*VOLUMEN, dtype=torch.float64)
    assert torch.equal(model.grad(x.clone()), model.grad(x.clone()))


def test_energy_devuelve_densidad_por_voxel(model):
    """`energy()` devuelve la densidad rho con forma [B,1,D,H,W], no un escalar por muestra."""
    x = torch.randn(*VOLUMEN, dtype=torch.float64)
    assert model.energy(x).shape == (2, 1, 13, 14, 15)


def test_no_bias_anywhere(model):
    """Bias-free de punta a punta: con la media cero de K1, la energía es invariante a offsets."""
    assert not [name for name, _ in model.named_parameters() if "bias" in name]
    for module in model.modules():
        if hasattr(module, "bias"):
            assert module.bias is None


def test_k1_del_modelo_sale_proyectado():
    """El K1 que arma `TotalDeepVariation3D`, tal como sale del constructor y sin re-proyectar:
    media cero y norma 1 por filtro, y `proj` lo devuelve a la bola unidad. La norma <= 1 es la
    que pide el análisis de estabilidad del paper; tests/test_conv3d.py prueba un Conv3d suelto.

    En float64 desde la construcción, para medir la proyección y no un casteo desde float32."""
    previo = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = TotalDeepVariation3D(dict(CONFIG))
    finally:
        torch.set_default_dtype(previo)
    assert model.K1.zero_mean and model.K1.bound_norm
    assert hasattr(model.K1.weight, "proj")

    w = model.K1.weight.data
    assert w.dtype == torch.float64
    means = w.sum(dim=(1, 2, 3, 4))
    norms = w.flatten(1).norm(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-12)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)

    model.K1.weight.data.mul_(50.0)
    model.K1.weight.proj()
    assert (model.K1.weight.data.flatten(1).norm(dim=1) <= 1.0 + 1e-12).all()


def test_invariancia_a_offsets(model):
    """Lo que el bias-free y la media cero de K1 compran: sumarle una constante al volumen no
    cambia el gradiente. Sin `zero_mean` en K1 se rompería.

    K1 se proyectó en float32 y quedó con una media de ~1e-8: se re-proyecta en doble precisión."""
    model.K1.weight.proj()
    torch.manual_seed(6)
    x = torch.randn(*VOLUMEN, dtype=torch.float64)
    assert torch.allclose(model.grad(x + 0.37), model.grad(x), atol=1e-12)


def test_parametros_tdv33_gris():
    tdv = TotalDeepVariation3D({"in_channels": 1, "num_features": 32, "num_scales": 3, "num_mb": 3,
                                "multiplier": 1})
    assert sum(p.numel() for p in tdv.parameters()) == PARAMETROS_TDV33_GRIS


# --- SEGUNDA DERIVADA ------------------------------------------------------------------------
# El backward del entrenamiento usa la DERIVADA de `grad()`: autograd atraviesa la cadena adjunta
# escrita a mano. Que la adjunta sea correcta (tests/test_conv3d.py) no implica que su derivada
# lo sea.

def test_hessiano_por_vector_matches_autograd(model):
    """`d(grad_R)/dx · v`, manual contra la referencia: el jacobiano de cada paso desenrollado."""
    torch.manual_seed(4)
    base = torch.randn(*VOLUMEN, dtype=torch.float64)
    v = torch.randn_like(base)

    xa = base.clone().requires_grad_(True)
    (manual,) = torch.autograd.grad((model.grad(xa) * v).sum(), xa)

    xb = base.clone().requires_grad_(True)
    (reference,) = torch.autograd.grad((model.grad_autograd_reference(xb) * v).sum(), xb)

    assert torch.allclose(manual, reference, atol=1e-9), (
        f"desviación máxima {(manual - reference).abs().max().item():.3e}"
    )


def test_segunda_derivada_en_los_pesos_matches_autograd(model):
    """`d(grad_R)/dtheta · v`: es lo que Adam aplica a cada peso."""
    torch.manual_seed(5)
    base = torch.randn(*VOLUMEN, dtype=torch.float64)
    v = torch.randn_like(base)
    pesos = [p for p in model.parameters() if p.requires_grad]

    xa = base.clone().requires_grad_(True)
    manual = torch.autograd.grad((model.grad(xa) * v).sum(), pesos)

    xb = base.clone().requires_grad_(True)
    reference = torch.autograd.grad((model.grad_autograd_reference(xb) * v).sum(), pesos)

    peor = max((m - r).abs().max().item() for m, r in zip(manual, reference))
    assert peor < 1e-9, f"desviación máxima {peor:.3e} en {len(pesos)} tensores"


# --- VNET3D ----------------------------------------------------------------------------------

def test_vnet3d_corre_s_pasos():
    """Forward de S pasos: devuelve [x_0, ..., x_S], x_0 es la entrada, y cada paso es
    x_s = prox(x_{s-1} − (T/S)·∇R(x_{s-1}), z, λ/S) con el regularizador 3D."""
    steps = 3
    torch.manual_seed(0)
    vnet = VNet3D(vnet_config(CONFIG, steps)).double()
    assert isinstance(vnet.R, TotalDeepVariation3D)

    torch.manual_seed(7)
    z = torch.rand(*VOLUMEN, dtype=torch.float64)
    with torch.no_grad():
        x_all = vnet(z, z)
        assert x_all.shape == (steps + 1, *VOLUMEN)
        assert torch.equal(x_all[0], z)

        x = z
        for s in range(1, steps + 1):
            x = vnet.D.prox(x - vnet.T / steps * vnet.R.grad(x), z, vnet.lmbda / steps)
            assert torch.allclose(x_all[s], x, atol=1e-12)
    # el esquema mueve la entrada: con T > 0 la salida no es z
    assert not torch.allclose(x_all[-1], z)


def test_vnet3d_checkpointing_es_exacto():
    """`efficient=True` recomputa el forward en el backward: en float64 sobre CPU el gradiente de
    la pérdida en T, lambda y los pesos debe ser el mismo que sin checkpointing."""
    torch.manual_seed(0)
    base = VNet3D(vnet_config(CONFIG, 2)).double()
    torch.manual_seed(8)
    z = torch.rand(*VOLUMEN, dtype=torch.float64)
    y = torch.rand(*VOLUMEN, dtype=torch.float64)

    grads = []
    for efficient in (False, True):
        vnet = copy.deepcopy(base)
        vnet.efficient = efficient
        x0 = z.clone().requires_grad_(True)
        loss = ((vnet(x0, z)[-1] - y) ** 2).sum()
        grads.append(torch.autograd.grad(loss, list(vnet.parameters())))

    peor = max((a - b).abs().max().item() for a, b in zip(*grads))
    assert peor < 1e-10, f"desviación máxima {peor:.3e}"


def test_vnet3d_tiene_las_claves_de_vnet2d():
    """Mismos nombres de atributo que en 2D: las claves del `state_dict` 3D son las del 2D sin
    los `mask` (que conv3d.py no tiene), y cada kernel tiene un eje espacial más."""
    torch.manual_seed(0)
    sd2 = VNet2D(vnet_config(CONFIG, 2)).state_dict()
    sd3 = VNet3D(vnet_config(CONFIG, 2)).state_dict()
    assert list(sd3) == [k for k in sd2 if not k.endswith(".mask")]
    for k in sd3:
        if sd2[k].dim() == 4:
            assert sd3[k].shape == (*sd2[k].shape, sd2[k].shape[-1])
        else:
            assert sd3[k].shape == sd2[k].shape


@pytest.mark.skipif(not CHECKPOINT_GRIS.is_file(), reason="requiere checkpoints/Oficiales")
def test_vnet2d_sigue_cargando_el_oficial():
    """`VNet2D.TDV` es atributo de clase, no submódulo: el `state_dict` 2D no cambia y el oficial
    carga con `strict=True` aunque falten las imágenes que pide test_reproduction.py."""
    ck = torch.load(CHECKPOINT_GRIS, map_location="cpu", weights_only=False)
    vnet = VNet2D(ck["config"])
    assert type(vnet.R) is TotalDeepVariation2D
    vnet.load_state_dict(ck["model"], strict=True)
