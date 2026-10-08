"""Regularizador Total Deep Variation (Kobler et al., 2020). Port de `../tdv/ddr/tdv.py`.

TDV no es un denoiser: es una energía aprendida R(x). La red emite una densidad escalar por píxel
y lo que consume el esquema es su **gradiente**.

    rho(x) = (1/m) * K_N . M_L . ... . M_1 . K_1 (x)        m = num_features
    R(x)   = sum_pixels rho(x)

`energy(x)` y `grad(x)` son APIs separadas a propósito. `grad` recorre el grafo al revés con las
adjuntas de `conv.py`; el entrenamiento retropropaga a través de él, así que la activación tiene
que ser C^2 (Student-t, nunca ReLU).

Diferencia con `../tdv/ddr/tdv.py`: allí `MicroBlock` guarda `act_prime` como estado mutable del
módulo, inseguro con checkpointing y llamadas repetidas. Acá se devuelve como caché local. Los
nombres de parámetros y buffers se preservan, así que el checkpoint oficial carga sin traducción.
"""


from __future__ import annotations

from typing import Any

import torch

from .conv import Conv2d, ConvScale2d, ConvScaleTranspose2d

__all__ = ["TotalDeepVariation2D", "student_t"]

# Cachés de derivadas de activación de un macro-bloque: por escala, una lista de tensores.
MacroCache = list[list[torch.Tensor]]


def student_t(x: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """phi(a) = log(1+alpha*a^2)/(2*alpha) y su derivada phi'(a) = a/(1+alpha*a^2).

    Se promueve a float32 si la entrada viene en half: las convoluciones pueden correr bajo
    autocast, la Student-t no. float64 se respeta para los tests de gradiente.
    """
    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    d = 1.0 + alpha * xf.square()
    return torch.log(d) / (2 * alpha), xf / d


class StudentT2(torch.nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return student_t(x, self.alpha)


class MicroBlock(torch.nn.Module):
    """Bloque residual bias-free `v = u + K2 phi(K1 u)`. `forward` devuelve `(v, act_prime)`;
    `backward` aplica J(u)^T q = q + K1^T [ phi'(K1 u) * K2^T q ] con esa caché."""

    def __init__(self, num_features: int, bound_norm: bool = False, invariant: bool = False) -> None:
        super().__init__()
        self.conv1 = Conv2d(
            num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False
        )
        self.act = StudentT2(alpha=1.0)
        self.conv2 = Conv2d(
            num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, act_prime = self.act(self.conv1(x))
        return x + self.conv2(a), act_prime

    def backward(self, grad_out: torch.Tensor, act_prime: torch.Tensor) -> torch.Tensor:
        return grad_out + self.conv1.backward(act_prime * self.conv2.backward(grad_out))


class MacroBlock(torch.nn.Module):
    """Bloque multiescala; con `num_scales=3` son cinco micro-bloques. El estado multiescala
    persiste entre macro-bloques: las escalas gruesas no se descartan."""

    def __init__(
        self,
        num_features: int,
        num_scales: int = 3,
        multiplier: int = 1,
        bound_norm: bool = False,
        invariant: bool = False,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales

        mb: list[torch.nn.ModuleList] = []
        for i in range(num_scales - 1):
            mb.append(
                torch.nn.ModuleList(
                    [
                        MicroBlock(num_features * multiplier**i, bound_norm=bound_norm, invariant=invariant),
                        MicroBlock(num_features * multiplier**i, bound_norm=bound_norm, invariant=invariant),
                    ]
                )
            )
        # la escala más gruesa tiene un solo micro-bloque
        mb.append(
            torch.nn.ModuleList(
                [
                    MicroBlock(
                        num_features * multiplier ** (num_scales - 1),
                        bound_norm=bound_norm,
                        invariant=invariant,
                    )
                ]
            )
        )
        self.mb = torch.nn.ModuleList(mb)

        conv_down, conv_up = [], []
        for i in range(1, num_scales):
            conv_down.append(
                ConvScale2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
            conv_up.append(
                ConvScaleTranspose2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
        self.conv_down = torch.nn.ModuleList(conv_down)
        self.conv_up = torch.nn.ModuleList(conv_up)

    def forward(self, x: list[torch.Tensor | None]) -> tuple[list[torch.Tensor], MacroCache]:
        assert len(x) == self.num_scales
        cache: MacroCache = [[] for _ in range(self.num_scales)]

        # bajada: extracción de features y submuestreo
        for i in range(self.num_scales - 1):
            x[i], ap = self.mb[i][0](x[i])
            cache[i].append(ap)
            x_down = self.conv_down[i](x[i])
            x[i + 1] = x_down if x[i + 1] is None else x[i + 1] + x_down

        # escala más gruesa: un solo micro-bloque
        last = self.num_scales - 1
        x[last], ap = self.mb[last][0](x[last])
        cache[last].append(ap)

        # subida
        for i in reversed(range(self.num_scales - 1)):
            x[i] = x[i] + self.conv_up[i](x[i + 1], x[i].shape)
            x[i], ap = self.mb[i][1](x[i])
            cache[i].append(ap)

        return x, cache  # type: ignore[return-value]

    def backward(
        self, grad_x: list[torch.Tensor | None], cache: MacroCache
    ) -> list[torch.Tensor]:
        # transpuesta de la subida
        for i in range(self.num_scales - 1):
            grad_x[i] = self.mb[i][1].backward(grad_x[i], cache[i][1])
            grad_up = self.conv_up[i].backward(grad_x[i])
            grad_x[i + 1] = grad_up if grad_x[i + 1] is None else grad_x[i + 1] + grad_up

        last = self.num_scales - 1
        grad_x[last] = self.mb[last][0].backward(grad_x[last], cache[last][0])

        # transpuesta de la bajada
        for i in reversed(range(self.num_scales - 1)):
            grad_x[i] = grad_x[i] + self.conv_down[i].backward(grad_x[i + 1], grad_x[i].shape)
            grad_x[i] = self.mb[i][0].backward(grad_x[i], cache[i][0])

        return grad_x  # type: ignore[return-value]


class TotalDeepVariation2D(torch.nn.Module):
    """TDV33 = `num_mb=3`, `num_scales=3`, `num_features=32`. El potencial es psi(x)=x
    (ablación del paper, Tabla 1)."""

    def __init__(self, config: dict[str, Any] | None = None, file: str | None = None) -> None:
        super().__init__()

        if (config is None) == (file is None):
            raise RuntimeError("especificar O BIEN un dict de config O BIEN un archivo `.pth`")

        state_dict = None
        if file is not None:
            if not file.endswith(".pth"):
                raise ValueError("el archivo debe terminar en `.pth`")
            checkpoint = torch.load(file, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            state_dict = checkpoint["model"]

        assert config is not None
        self.in_channels: int = config["in_channels"]
        self.num_features: int = config["num_features"]
        self.multiplier: int = config["multiplier"]
        self.num_mb: int = config["num_mb"]
        self.zero_mean: bool = config.get("zero_mean", True)
        self.num_scales: int = config.get("num_scales", 3)

        # K1: análisis, bias-free, media cero y norma l2 <= 1 por filtro
        self.K1 = Conv2d(
            self.in_channels,
            self.num_features,
            3,
            zero_mean=self.zero_mean,
            invariant=False,
            bound_norm=True,
            bias=False,
        )

        self.mb = torch.nn.ModuleList(
            [
                MacroBlock(
                    self.num_features,
                    num_scales=self.num_scales,
                    bound_norm=False,
                    invariant=False,
                    multiplier=self.multiplier,
                )
                for _ in range(self.num_mb)
            ]
        )

        # K_N: cabeza 1x1 de un canal, bias-free
        self.KN = Conv2d(self.num_features, 1, 1, invariant=False, bound_norm=False, bias=False)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    # -- transformación y su transpuesta ------------------------------------------------

    def _transformation(self, x: torch.Tensor) -> tuple[torch.Tensor, list[MacroCache]]:
        x = self.K1(x)
        state: list[torch.Tensor | None] = [x] + [None] * (self.num_scales - 1)
        caches: list[MacroCache] = []
        for i in range(self.num_mb):
            state, cache = self.mb[i](state)
            caches.append(cache)
        return self.KN(state[0]), caches

    def _transformation_T(self, grad_out: torch.Tensor, caches: list[MacroCache]) -> torch.Tensor:
        grad = self.KN.backward(grad_out)
        state: list[torch.Tensor | None] = [grad] + [None] * (self.num_scales - 1)
        for i in reversed(range(self.num_mb)):
            state = self.mb[i].backward(state, caches[i])
        return self.K1.backward(state[0])

    def _potential(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.num_features

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        # psi'(x) = 1, escalado por el número de features
        return torch.ones_like(x) / self.num_features

    # -- API pública ---------------------------------------------------------------------

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Densidad de energía por píxel, con forma `[B, 1, H, W]`."""
        out, _ = self._transformation(x)
        return self._potential(out)

    def grad(self, x: torch.Tensor, get_energy: bool = False):
        """Gradiente exacto de la energía sumada. Es lo que consume el esquema."""
        out, caches = self._transformation(x)
        energy = self._potential(out) if get_energy else None
        grad = self._transformation_T(self._activation(out), caches)
        return (energy, grad) if get_energy else grad

    def grad_autograd_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Referencia lenta para tests. No usar en producción."""
        x = x.requires_grad_(True)
        energy = self.energy(x).sum()
        (grad,) = torch.autograd.grad(energy, x, create_graph=True)
        return grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad(x)
