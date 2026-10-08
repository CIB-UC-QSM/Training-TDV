"""Tarjetas de fórmulas del entrenamiento de TDV, una PNG por idea, para pegar en diapositivas.

    uv run python docs/presentacion/formulas.py      # salida/formula_*.png, unos segundos

Cada fórmula sale del código, no del paper: el puntero de la derecha dice dónde está
implementada. Mathtext de matplotlib (no hay LaTeX en esta máquina); los strings se verificaron
uno por uno contra el parser.
"""
import numpy as np

from estilo import C, plt, guardar

plt.rcParams.update({"mathtext.fontset": "cm", "axes.grid": False})
TINTA, GRIS, ACENTO = "#1d1c1a", "#5b5a55", C["final"]

TARJETAS = {
    "formula_1_problema": (
        "El problema: sacar el ruido",
        [r"$z = y + \frac{\sigma}{255}\,\xi,\qquad \xi \sim \mathcal{N}(0, I),\qquad \sigma = 25$"],
        ["y : imagen limpia en [0, 1]      z : imagen con ruido (lo único que se ve)",
         "El ruido ξ se vuelve a sortear en cada iteración de entrenamiento"],
        "ddr/data.py:94-98"),
    "formula_2_energia": (
        "TDV: una energía con un término de dato y uno aprendido",
        [r"$E(x) = \frac{\lambda}{T}\,\frac{1}{2}\Vert x - z \Vert^2 + R(x;\theta)$"],
        ["1er término: parecerse al dato z      2º término: parecer una imagen natural",
         "R es una red (≈388 000 pesos θ) que devuelve un número: cuán «poco natural» es x",
         "λ pesa el dato y T es cuánto tiempo se baja por la energía: dos escalares aprendidos"],
        "ddr/model.py:79-102"),
    "formula_3_regularizador": (
        "El regularizador R: convolución, tres U-Nets, convolución",
        [r"$R(x;\theta) = \sum_{p} \frac{1}{m}\left[K_N\,(M_3 \circ M_2 \circ M_1)(K_1 x)\right]_p$",
         r"$\mathrm{bloque\ residual:}\quad v = u + K_2\,\phi(K_1 u),\qquad"
         r" \phi(a) = \frac{1}{2}\log(1 + a^2)$"],
        ["K₁ : 32 filtros 3×3 sobre RGB      M₁, M₂, M₃ : macro-bloques (U de 3 escalas)",
         "K_N : de 32 canales a 1      sin bias en ninguna capa      m = 32",
         "φ es suave (C²): hace falta porque el entrenamiento deriva a través de ∇R"],
        "ddr/tdv.py:33-69, 210-276"),
    "formula_4_esquema": (
        "Inferencia: S = 10 pasos de descenso sobre E",
        [r"$x_{s+1} = \frac{x_s - \frac{T}{S}\,\nabla R(x_s) + \frac{\lambda}{S}\,z}{1 + \frac{\lambda}{S}},"
         r"\qquad s = 0, \ldots, S-1,\qquad x_0 = z$"],
        ["Un paso de gradiente sobre R de tamaño T/S, y después el prox (cerrado) del término de dato",
         "La salida es x_S: diez pasos, siempre los mismos, sin criterio de parada"],
        "ddr/model.py:35-36, 79-102 · train.py:42"),
    "formula_5_perdida": (
        "Entrenamiento: que x_S se parezca a la imagen limpia",
        [r"$\mathcal{L}(\Theta) = \frac{1}{B}\sum_{b=1}^{B} \Vert x_S^{(b)} - y^{(b)} \Vert^2,\qquad B = 32$",
         r"$\Theta = \{K_1, M_1, M_2, M_3, K_N, T, \lambda\}:\quad 387\,970$ parámetros"],
        ["Parches de 93×93 de BSDS400; la pérdida mira sólo el último estado x_S",
         "El gradiente atraviesa los 10 pasos hacia atrás (el «adjunto»): ahí nacen los estallidos"],
        "train.py:35-47, 255-265"),
    "formula_6_adam": (
        "Adam y el desplazamiento δ que deja un pico",
        [r"$\Theta_t = \Theta_{t-1} - \eta_t\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},\qquad"
         r" \eta_t = 4\cdot 10^{-4}\cdot 0.5^{\lfloor t/25000 \rfloor}$",
         r"$\delta_{\mathrm{Adam}} = \frac{\eta}{\sqrt{1-\beta_2}} = \frac{4\cdot 10^{-4}}{\sqrt{0.001}} \approx 0.0126$"],
        ["Un pico de gradiente, por grande que sea, mueve cada parámetro ≈ δ (régimen estacionario)",
         "Por eso el tamaño del pico no importa (y el clipping no ayuda): importa cuánto margen hay",
         "El lr se divide por 2 cada 25 000 iteraciones, y δ con él"],
        "train.py:38-40, 51, 214-215"),
    "formula_7_pared": (
        "La «pared» de T: medirlo en pasos de Adam",
        [r"$\frac{T}{\delta(t)} < 1\ \ \Longrightarrow\ \ \mathrm{un\ solo\ pico\ puede\ llevar}\ T \to 0$",
         r"$T_0 = 0.001 \approx 0.08\,\delta\ \ (\mathrm{receta\ vieja})\qquad"
         r"T_0 = 8\,\delta \approx 0.101\ \ (\mathrm{las\ finales})$"],
        ["Con T = 0 la salida es la imagen ruidosa (20.17 dB) y los pesos dejan de recibir gradiente;",
         "v5 quedó así 97 500 iteraciones (v1 salió un rato y volvió a morir). No es absorbente: decide el signo de ∂L/∂T",
         "En las finales (T₀ = 8δ), T no bajó de 6δ y con el lr ÷ 2 llegó a 25δ"],
        "train.py:51-52 · docs/bitacora.md:508-521"),
    "formula_8_disyuntor": (
        "El disyuntor: rechazar el lote y, si no alcanza, rebobinar",
        [r"$\mathrm{rechazar\ si}\quad \Vert g_t\Vert > 5\cdot\mathrm{mediana}_{100}\Vert g\Vert"
         r"\quad \vee \quad \Vert g_t\Vert > 10^4$",
         r"$25\ \mathrm{rechazos\ seguidos}\ \Longrightarrow\ (\theta, m, v) \leftarrow "
         r"(\theta, m, v)_{\mathrm{respaldo}}$"],
        ["Rechazar = no dar el paso de Adam, así m y v no se contaminan con el pico",
         "El respaldo se toma cada 250 iteraciones sanas, con 3 limpias seguidas",
         "Lo que salva la corrida es el rebobinado: el rechazo sólo contiene la secuela"],
        "train.py:60-66, 100-136"),
    "formula_9_psnr": (
        "La métrica: PSNR promedio en CBSD68",
        [r"$\mathrm{PSNR} = 10\log_{10}\frac{1}{\mathrm{MSE}(\hat{x}, y)},\qquad"
         r" \overline{\mathrm{PSNR}} = \frac{1}{68}\sum_{i=1}^{68}\mathrm{PSNR}_i$"],
        ["68 imágenes que no se usan para entrenar, σ = 25, una realización de ruido fija por imagen",
         "Meta: 31.53 dB (paper); el checkpoint oficial da 31.534 con este código.  +0.1 dB ya se nota"],
        "ddr/data.py:101-107 · evaluate.py:37-59"),
}


def tarjeta(nombre, titulo, formulas, notas, fuente):
    alto = 1.55 + 0.95 * len(formulas) + 0.42 * len(notas)
    fig = plt.figure(figsize=(12, alto))
    y = 1 - 0.55 / alto
    fig.text(0.02, y, titulo, fontsize=21, fontweight="bold", color=TINTA, va="top")
    y -= 0.75 / alto
    for f in formulas:
        fig.text(0.04, y, f, fontsize=27, color=TINTA, va="top")
        y -= 0.95 / alto
    y -= 0.05 / alto
    for n in notas:
        fig.text(0.04, y, n, fontsize=15, color=GRIS, va="top")
        y -= 0.42 / alto
    fig.text(0.98, 0.1 / alto, "implementado en " + fuente, fontsize=11, color="#8c8a83",
             ha="right", va="bottom", family="monospace")
    fig.add_artist(plt.Line2D([0.02, 0.02], [y + 0.3 / alto, 1 - 1.1 / alto], color=ACENTO, lw=4))
    guardar(fig, nombre)


def figura_phi():
    """φ y φ': la no linealidad de cada bloque, suave y con derivada acotada."""
    a = np.linspace(-6, 6, 600)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(a, 0.5 * np.log1p(a ** 2), color=C["final"], lw=2.6)
    axs[0].set_title(r"$\phi(a) = \frac{1}{2}\log(1+a^2)$", fontsize=18)
    axs[1].plot(a, a / (1 + a ** 2), color=C["fallida"], lw=2.6)
    axs[1].set_title(r"$\phi'(a) = \frac{a}{1+a^2}$  (acotada en ±½)", fontsize=18)
    for ax in axs:
        ax.axhline(0, color="#bdbbb3", lw=0.8); ax.axvline(0, color="#bdbbb3", lw=0.8)
        ax.grid(True); ax.set_xlabel("a")
    guardar(fig, "formula_phi")


if __name__ == "__main__":
    for n, t in TARJETAS.items():
        tarjeta(n, *t)
    figura_phi()
