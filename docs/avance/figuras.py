"""Figuras para la presentación de avance. Uso: uv run python figuras.py (desde Training-TDV)."""
import json, re, glob, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "runs"
OUT = "docs/avance/salida"
C = dict(v4="#2a78d6", ref="#eb6834", vivas="#1baf7a", muertas="#e34948", v1="#4a3aa7",
         v5="#eda100", orac="#52514e")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "figure.dpi": 130, "savefig.bbox": "tight"})
ORAC_PSNR, T_STAR, L_STAR = 31.534, 0.024727, 0.104522


def parse_log(path):
    it, T, lam, g, ev = [], [], [], [], []
    last = 0
    for line in open(path):
        m = re.match(r"\[\s*(\d+)/\d+\] loss=\S+ lr=\S+ T=(\S+) lambda=(\S+)(?: \|g\|=(\S+))?", line)
        if m:
            last = int(m[1]); it.append(last); T.append(float(m[2])); lam.append(float(m[3]))
            g.append(float(m[4]) if m[4] else np.nan)
        m = re.search(r"eval PSNR=(\S+) dB", line)
        if m:
            ev.append((last, float(m[1])))
    return dict(it=np.array(it), T=np.array(T), lam=np.array(lam), g=np.array(g),
                ev=np.array(ev))


def hist(d):
    h = json.load(open(f"{d}/history.json"))
    return np.array([[r["iteration"], r["psnr"]] for r in h])


v4 = parse_log(f"{R}/replica-v4/train.log")
v1 = parse_log(f"{R}/replica-bf16/train.log")
v5 = parse_log(f"{R}/replica-v5/train.log")
ref_tr = np.load(f"{R}/referencia-100k/traza.npy")      # loss T dT lambda dlambda |g|
ref_h = hist(f"{R}/referencia-100k")
cortas = []
for d in sorted(glob.glob(f"{R}/precision/*-seed*")):
    tr, h = np.load(f"{d}/traza.npy"), hist(d)
    viva = np.all(np.isfinite(tr[:, 5])) and tr[:, 5].max() < 1e4
    cortas.append(dict(name=d.split("/")[-1], tr=tr, h=h, viva=viva))
print("cortas vivas (max|g|<1e4):", sum(c["viva"] for c in cortas), "/", len(cortas))

# 1. PSNR en toda la corrida
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(ORAC_PSNR, color=C["orac"], ls="--", lw=1)
ax.text(100, ORAC_PSNR + 0.08, "paper / checkpoint oficial 31.53 dB", color=C["orac"], fontsize=9)
ax.plot(v4["ev"][:, 0], v4["ev"][:, 1], "-o", ms=3, color=C["v4"], label="v4 (ago-sep, receta vieja, bf16 + clip)")
ax.plot(v1["ev"][:, 0], v1["ev"][:, 1], "-o", ms=3, color=C["v1"], label="v1 (receta vieja, bf16)")
ax.plot(ref_h[:, 0], ref_h[:, 1], "-o", ms=3, color=C["ref"], label="referencia-100k (actual, fp32)")
for c in cortas:
    ax.plot(c["h"][:, 0], c["h"][:, 1], "-", lw=0.8, alpha=0.6,
            color=C["vivas"] if c["viva"] else C["muertas"])
ax.plot([], [], color=C["vivas"], lw=0.8, label="24 corridas cortas actuales: sobreviven")
ax.plot([], [], color=C["muertas"], lw=0.8, label="24 corridas cortas actuales: estallan")
ax.set_ylim(28, 32); ax.set_xlim(0, 100000)
ax.set_xlabel("iteración"); ax.set_ylabel("PSNR en CBSD68 (dB)")
ax.set_title("PSNR durante el entrenamiento (fuera del rango = estallido)", loc="left")
ax.legend(fontsize=8.5, loc="lower right", frameon=False)
fig.savefig(f"{OUT}/f1_psnr_todo.png")

# 2. Zoom 0-4000: misma curva
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for c in cortas:
    ax.plot(c["h"][:, 0], c["h"][:, 1], "-o", ms=3, lw=1, alpha=0.7,
            color=C["vivas"] if c["viva"] else C["muertas"])
e = v4["ev"][v4["ev"][:, 0] <= 4000]
ax.plot(e[:, 0], e[:, 1], "-s", ms=7, lw=2.5, color=C["v4"], label="v4 en las mismas iteraciones", zorder=5)
ax.plot([], [], "-o", color=C["vivas"], label="actuales que sobreviven")
ax.plot([], [], "-o", color=C["muertas"], label="actuales que estallan")
ax.set_ylim(30.4, 31.4); ax.set_xticks([1000, 2000, 3000, 4000])
ax.set_xlabel("iteración"); ax.set_ylabel("PSNR (dB)")
ax.set_title("Hasta la it 4 000: v4 y las corridas actuales son la misma curva", loc="left")
ax.legend(fontsize=9, frameon=False, loc="lower right")
fig.savefig(f"{OUT}/f2_psnr_zoom4k.png")

# 3. T y λ en el tiempo (dos paneles, no doble eje)
it_ref = np.arange(1, len(ref_tr) + 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, col, star, name in ((axs[0], "T", 1, T_STAR, "T (tiempo de parada)"),
                                 (axs[1], "lam", 3, L_STAR, "λ (peso del dato)")):
    ax.plot(v4["it"], v4[key], color=C["v4"], label="v4")
    ax.plot(it_ref[::50], ref_tr[::50, col], color=C["ref"], label="referencia-100k")
    for c in cortas[:6]:
        ax.plot(np.arange(1, len(c["tr"]) + 1)[::20], c["tr"][::20, col], lw=0.8, alpha=0.7,
                color=C["vivas"] if c["viva"] else C["muertas"])
    ax.axhline(star, color=C["orac"], ls="--", lw=1)
    ax.text(60000, star * 1.08, "checkpoint oficial", color=C["orac"], fontsize=8.5)
    ax.set_yscale("log"); ax.set_xlabel("iteración"); ax.set_title(name, loc="left")
axs[1].set_ylim(1e-4, 1); axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("Escalares aprendidos en el tiempo  (verde/rojo: 6 corridas cortas actuales)", x=0.01, ha="left")
fig.savefig(f"{OUT}/f3_T_lambda.png")

# 4. Plano de fase (T, λ) con flechas
fig, ax = plt.subplots(figsize=(7, 5.6))
def tray(T, L, color, label, n=14, lw=1.8):
    ax.plot(T, L, color=color, lw=lw, label=label, alpha=0.9)
    idx = np.linspace(0, len(T) - 2, n).astype(int)
    for i in idx:
        ax.annotate("", xy=(T[i + 1], L[i + 1]), xytext=(T[i], L[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12))
    ax.plot(T[0], L[0], "o", color=color, ms=7); ax.plot(T[-1], L[-1], "X", color=color, ms=9)
m = v4["lam"] > 0
tray(v4["T"][m], v4["lam"][m], C["v4"], "v4 (68k it buenas, luego estallido)")
tray(ref_tr[::100, 1], ref_tr[::100, 3], C["ref"], "referencia-100k")
for c in cortas[:12]:
    if c["name"].startswith("rapido"):
        tray(c["tr"][::40, 1], c["tr"][::40, 3], C["vivas"] if c["viva"] else C["muertas"], None, n=3, lw=0.8)
ax.plot(T_STAR, L_STAR, "*", ms=18, color="#0b0b0b", label="checkpoint oficial")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-4, 0.6)
ax.set_xlabel("T"); ax.set_ylabel("λ")
ax.set_title("Trayectoria de (T, λ) durante el entrenamiento\n● inicio   ✕ final", loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.savefig(f"{OUT}/f4_fase_T_lambda.png")

# 5. ‖g‖ en el tiempo
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
axs[0].semilogy(v4["it"], v4["g"], color=C["v4"], lw=1)
axs[0].set_title("v4: ‖g‖ cada 250 it", loc="left")
axs[1].semilogy(it_ref[::25], ref_tr[::25, 5], color=C["ref"], lw=0.5)
axs[1].set_title("referencia-100k: ‖g‖ por iteración", loc="left")
for ax in axs:
    ax.axhline(1e4, color=C["muertas"], ls="--", lw=1)
    ax.text(2000, 2e4, "umbral de evento 10⁴", color=C["muertas"], fontsize=8.5)
    ax.set_xlabel("iteración")
axs[0].set_ylabel("norma del gradiente")
fig.savefig(f"{OUT}/f5_grad.png")

# 6. Filtros K1 (32 de 3x3x3, como RGB)
sys.path.insert(0, ".")
from evaluate import load_vnet
from ddr.model import VNet2D
ck = torch.load("checkpoints/Oficiales/tdv3-3-25-f32-color.pth",
                map_location="cpu", weights_only=False)
torch.manual_seed(0)
init = VNet2D(ck["config"]).R.K1.weight.detach()
fuentes = [("init aleatoria (it 0)", init),
           ("actual, it 4 000", load_vnet(f"{R}/precision/rapido-seed0/best.pth")[0].R.K1.weight.detach()),
           ("v4, it 68 000", load_vnet(f"{R}/replica-v4/checkpoint.pth")[0].R.K1.weight.detach()),
           ("checkpoint oficial", ck["model"]["R.K1.weight"])]
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (name, w) in zip(axs, fuentes):
    w = w.numpy()
    w = w - w.mean(axis=(1, 2, 3), keepdims=True)
    w = w / (np.abs(w).max(axis=(1, 2, 3), keepdims=True) + 1e-12) * 0.5 + 0.5
    grid = np.ones((4 * 4 + 1, 8 * 4 + 1, 3))
    for k in range(32):
        r, cc = divmod(k, 8)
        grid[1 + r * 4:1 + r * 4 + 3, 1 + cc * 4:1 + cc * 4 + 3] = w[k].transpose(1, 2, 0)
    ax.imshow(grid, interpolation="nearest"); ax.set_axis_off(); ax.set_title(name, fontsize=10)
fig.suptitle("Primera capa K1: 32 filtros 3×3 RGB (cada uno normalizado)", x=0.01, ha="left")
fig.savefig(f"{OUT}/f6_filtros_K1.png")

# similitud de K1 entre modelos (coseno de filtros, emparejados por permutación óptima)
from scipy.optimize import linear_sum_assignment
def sim(a, b):
    a = a.reshape(32, -1).numpy(); b = b.reshape(32, -1).numpy()
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    M = np.abs(a @ b.T); r, c = linear_sum_assignment(-M); return M[r, c].mean()
for n, w in fuentes:
    print(f"|cos| medio K1 vs oficial  {n:24s} {sim(w, fuentes[3][1]):.3f}")
print("ok")

# 7. Movimiento por capa desde el init
mv = json.load(open(f"{OUT}/movimiento.json"))
keys = list(mv["v4_68k"])
fig, ax = plt.subplots(figsize=(10, 4.2))
for tag, col, name in (("actual_4k", C["vivas"], "actual, it 4 000"),
                       ("ref100k_38k", C["ref"], "referencia-100k, it 38 000"),
                       ("v4_68k", C["v4"], "v4, it 68 000")):
    ax.plot(range(len(keys)), [mv[tag][k] for k in keys], "-o", ms=3.5, color=col, label=name)
ax.axhline(np.sqrt(2), color=C["orac"], ls="--", lw=1)
ax.text(0.5, 1.45, "√2 ≈ distancia entre dos inits independientes (el oficial está a 1.4–2.1)",
        color=C["orac"], fontsize=8.5)
lab = ["K1"] + ["" for _ in keys[1:-1]] + ["KN"]
for i, k in enumerate(keys):
    if k.endswith("conv_down.0.weight"):
        lab[i] = "macro " + k.split(".")[2]
ax.set_xticks(range(len(keys))); ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylabel("‖W − W₀‖ / ‖W₀‖"); ax.set_xlabel("capa (de entrada a salida)")
ax.set_title("Cuánto se alejó cada capa de su inicialización", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(f"{OUT}/f7_movimiento_capas.png")

# 8. Los vectores detrás de las figuras: trayectorias de T, λ y ‖g‖
np.savez_compressed(f"{OUT}/trayectorias.npz",
                    v4_it=v4["it"], v4_T=v4["T"], v4_lambda=v4["lam"], v4_g=v4["g"], v4_psnr=v4["ev"],
                    v1_it=v1["it"], v1_T=v1["T"], v1_lambda=v1["lam"], v1_psnr=v1["ev"],
                    ref100k_traza=ref_tr.astype(np.float32), ref100k_psnr=ref_h,
                    **{f"{c['name']}_traza": c["tr"].astype(np.float32) for c in cortas},
                    **{f"{c['name']}_psnr": c["h"] for c in cortas})
print("trayectorias.npz")
