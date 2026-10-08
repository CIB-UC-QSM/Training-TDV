"""Eigenfunciones no lineales de R (∇R(x) = Λx), como la figura del paper, en dos láminas.

    uv run python docs/presentacion/fig_eigen.py

e1_eigen_comparacion: deseado contra indeseado, el estado final del flujo (1 000 iteraciones)
por modelo y por imagen. g6_eigen_trayectorias: los estados guardados de cada trayectoria real
(trayectorias.py), en orden: qué estructuras va aprendiendo R y cómo se deshacen al estallar.

Los .npz salen de `eigen_lote.sh` / `eigen_extra.sh`; la columna de un modelo que todavía no está
calculado se dibuja vacía con «falta», para que la lámina se pueda armar a medias.
"""
import numpy as np

from estilo import C, SALIDA, plt, guardar

EIG = SALIDA / "eig"
IMAGENES = [("", "imagen 1"), ("_img2", "imagen 2")]


def carga(tag):
    p = EIG / f"eig_{tag}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return dict(x=np.clip(d["it1000"], 0, 1), inicio=d["start"], lam=d["log"][-1, 2],
                residuo=d["log"][-1, 1])


def lamina(columnas, nombre, titulo):
    fig, axs = plt.subplots(len(IMAGENES), len(columnas) + 1,
                            figsize=(2.35 * (len(columnas) + 1), 2.75 * len(IMAGENES)),
                            gridspec_kw=dict(wspace=0.06, hspace=0.32))
    for r, (suf, etq_img) in enumerate(IMAGENES):
        ref = next((carga(t + suf) for t, *_ in columnas if carga(t + suf)), None)
        ax = axs[r, 0]
        if ref is not None:
            ax.imshow(ref["inicio"])
        ax.set_title("punto de partida" if r == 0 else "", fontsize=11.5, loc="center")
        ax.set_ylabel(etq_img, fontsize=12)
        for c, (tag, etq, col) in enumerate(columnas, start=1):
            ax = axs[r, c]
            d = carga(tag + suf)
            if d is None:
                ax.text(0.5, 0.5, "falta", ha="center", va="center", color=C["texto"],
                        transform=ax.transAxes, fontsize=12)
            else:
                ax.imshow(d["x"])
                # Residuo alto = el flujo no convergió: lo que se ve no es una eigenfunción.
                conv = "" if d["residuo"] < 0.1 else "  · no converge"
                ax.set_xlabel(f"Λ = {d['lam']:.3g}{conv}", fontsize=11, color="#2b2a27")
            if r == 0:
                ax.set_title(etq, fontsize=11.5, loc="center", color="#2b2a27")
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(3)
        for ax in axs[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=16, fontweight="bold")
    guardar(fig, nombre)


def trayectorias():
    """g6: una fila por trayectoria REAL (misma semilla, mismo código), estados en orden temporal.

    Sin los init: R recién inicializado es casi nulo (Λ ≈ −2·10⁻⁴) y cualquier imagen es
    eigenfunción; la primera columna es el recorte de partida, común a todas las filas.
    """
    from trayectorias import ESTADOS, TRAYECTORIAS, ALIAS_EIG
    filas = [(n, col, [t for t in ts if not t.startswith("init")]) for n, col, ts in TRAYECTORIAS]
    filas.append(("checkpoint oficial", C["oficial"], ["oficial"]))
    ncol = max(len(ts) for *_, ts in filas)
    fig, axs = plt.subplots(len(filas), ncol + 1, figsize=(2.5 * (ncol + 1) + 2.6, 2.7 * len(filas)),
                            gridspec_kw=dict(wspace=0.06, hspace=0.38))
    inicio = carga("oficial")["inicio"]
    for r, (nombre, col, ts) in enumerate(filas):
        axs[r, 0].imshow(inicio)
        if r == 0:
            axs[r, 0].set_title("punto de partida", fontsize=11.5, loc="center")
        axs[r, 0].set_ylabel(nombre.replace(" · ", "\n"), fontsize=10.5, rotation=0, ha="right",
                             va="center", labelpad=8)
        for c in range(ncol):
            ax = axs[r, c + 1]
            if c >= len(ts):
                ax.set_axis_off(); continue
            t = ts[c]
            d = carga(ALIAS_EIG.get(t, t))
            if d is None:
                ax.text(0.5, 0.5, "falta", ha="center", va="center", transform=ax.transAxes,
                        color=C["texto"], fontsize=12)
            else:
                ax.imshow(d["x"])
                conv = "" if d["residuo"] < 0.1 else " · no converge"
                ax.set_xlabel(f"Λ = {d['lam']:.3g}{conv}", fontsize=10.5)
            ax.set_title(ESTADOS[t][0], fontsize=11.5, loc="center")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3)
        for ax in axs[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Qué aprende R y cómo se rompe: eigenfunciones (∇R(x) = Λx) a lo largo de cada "
                 "trayectoria", x=0.01, ha="left", fontsize=16, fontweight="bold")
    guardar(fig, "g6_eigen_trayectorias")


if __name__ == "__main__":
    lamina([("oficial", "checkpoint oficial", C["oficial"]),
            ("v4", "v4 · it 68k", C["v4"]),
            ("final_s0", "final s0", C["final"]),
            ("final_s1", "final s1", C["final"]),
            ("ref100k", "fallida\nreferencia-100k", C["fallida"]),
            ("estallada", "estallada\n(−2.6 dB)", C["fallida"])],
           "e1_eigen_comparacion",
           "Qué estructuras prefiere R:  ∇R(x) = Λx tras 1 000 pasos de flujo")
    trayectorias()
