"""Red variacional: el esquema desenrollado donde vive el regularizador TDV.

    x_{s+1} = prox_D( x_s - (T/S) grad_R(x_s),  z,  lambda/S )

`T` (tiempo de parada) y `lambda` (peso del término de datos) son **controles aprendidos**, no
hiperparámetros. La ec. (7) del paper acopla ambos al mismo `T/S`; el código de los autores los
desacopla en dos escalares independientes y esta implementación conserva eso. Ojo: que los
checkpoints oficiales tengan `lambda != T` **no prueba nada**, porque `lambda/T` no es invariante de
gauge — transportados, son la ec. (7). Atarlos está medido y descartado: ver docs/bitacora.md.

`T` está acoplado a `S`: cambiar `S` en inferencia sin reajustar `T` degrada el resultado.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.checkpoint as cp

from .tdv import TotalDeepVariation2D

__all__ = ["VNet2D", "L2DenoiseDataterm"]


class L2DenoiseDataterm(torch.nn.Module):
    """Fidelidad `0.5*||x - z||^2` — denoising gaussiano aditivo, A = Id."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()

    def energy(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x - z) ** 2

    def prox(self, x: torch.Tensor, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return (x + tau * z) / (1 + tau)

    def grad(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return x - z


class VNet2D(torch.nn.Module):
    """Esquema desenrollado de S pasos con regularizador TDV."""

    def __init__(self, config: dict[str, Any], efficient: bool = False) -> None:
        super().__init__()

        self.efficient = efficient
        self.S: int = config["S"]

        for name, mode, spec in (
            ("T", config["T_mode"], config["T"]),
            ("lmbda", config["lambda_mode"], config["lambda"]),
        ):
            value = self._scalar(mode, spec)
            if isinstance(value, torch.nn.Parameter):
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)

        self.R = TotalDeepVariation2D(config["R"]["config"])
        self.D = L2DenoiseDataterm()
        self.use_prox: bool = config["D"]["config"]["use_prox"]

    @staticmethod
    def _scalar(mode: str, spec: dict[str, float]):
        """Escalar fijo o aprendido. Los aprendidos llevan `proj()`, la restricción de
        positividad que `project_()` re-aplica tras cada paso del optimizador."""
        value = torch.tensor(float(spec["init"]))
        if mode == "fixed":
            return value
        if mode != "learned":
            raise ValueError(f"modo desconocido: {mode!r}")
        p = torch.nn.Parameter(value)
        lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1000.0))
        p.proj = lambda: p.data.clamp_(lo, hi)
        return p

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Devuelve todos los estados `[x_0, ..., x_S]` apilados en la dimensión 0.

        La pérdida se evalúa sólo en `x_S`; los intermedios sirven para diagnóstico.
        """
        x_all = x.new_empty((self.S + 1, *x.shape))
        x_all[0] = x
        tau = self.T / self.S

        for s in range(1, self.S + 1):
            x = x.contiguous()
            # El checkpointing recomputa el forward durante el backward: exige determinismo.
            if self.efficient and x.requires_grad:
                grad_R = cp.checkpoint(self.R.grad, x, use_reentrant=False)
            else:
                grad_R = self.R.grad(x)

            if self.use_prox:
                x = self.D.prox(x - tau * grad_R.contiguous(), z, self.lmbda / self.S)
            else:
                x = x - tau * grad_R - (self.lmbda / self.S) * self.D.grad(x, z)
            x_all[s] = x

        return x_all

    def project_(self) -> None:
        """Media cero y norma <= 1 por filtro en K1, positividad de T y lambda.

        Llamar **después de cada `optimizer.step()`**: es una restricción del análisis de
        estabilidad del paper, no una preferencia. Omitirla entrena otro modelo.
        """
        with torch.no_grad():
            for p in self.parameters():
                if hasattr(p, "proj"):
                    p.proj()

    def extra_repr(self) -> str:
        return f"S={self.S}"
