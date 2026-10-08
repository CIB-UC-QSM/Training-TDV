"""Los números de las corridas finales, en una tabla, y las predicciones registradas contra ellos.

    uv run python docs/presentacion/numeros.py       # escribe salida/numeros.md

Para mañana: es lo que hay que correr para que el guion y el brief tengan los números finales.
Mientras alguna corrida no tenga `estado.json` (terminada o cortada), sale marcado PROVISIONAL.
La s0 se cortó a mano en la 66k (commit 5419800): las predicciones de la it 100 000 (1, E1, E5)
se leen sólo sobre las que llegaron.

Las predicciones son las de `docs/bitacora.md` (pre-registro de la tanda t8delta-100k y su
enmienda del 2026-10-05, líneas ~3577-3601 y ~3687-3714): se copian acá con su banda, no se
reinterpretan. La 1 es la ENMENDADA ([0.050, 0.075]); la original [0.018, 0.032] ya estaba
falsificada antes de terminar y se lista para que no se pierda.
"""
import json

import numpy as np

from estilo import SALIDA, ORAC_PSNR, DELTA_ADAM, UMBRAL_G, lee_corrida, terminada, cortada, estado_txt

FINALES = ["t8delta-100k-s0", "t8delta-100k-s1"]


def fila(n):
    r = lee_corrida(n)
    p = r["psnr"]
    b = p[np.nanargmax(p[:, 1])]
    rech = r.get("rech", np.zeros(len(r["it"]), bool))
    racha, maxracha = 0, 0
    for x in rech:
        racha = racha + 1 if x else 0
        maxracha = max(maxracha, racha)
    cen = r["traza"][:, 6]
    # El relativo se arma recién con 100 normas aceptadas (train.py:110-115); antes, un lote con
    # ‖g‖ > 1e4 lo agarra sólo la cota aunque traza[:, 6] ya muestre una mediana parcial.
    acep_previas = np.cumsum(~rech) - (~rech)
    with np.errstate(invalid="ignore", divide="ignore"):
        sola = int(((r["g"] > UMBRAL_G) & ((acep_previas < 100) | (r["g"] <= 5 * cen))).sum())
    # E6: la primera ‖g‖ después de cada rebobinado contra la mediana restaurada.
    e6 = [float(r["g"][b] / cen[b]) for b in np.flatnonzero(r["reb"]) + 1 if b < len(cen)]
    # E3/E4: la corrida cortada por el corte de luz es la misma trayectoria hasta la it 18 000.
    corte = lee_corrida(n.replace("t8delta-100k", "t8delta-100k-corte"))
    k = min(18_000, len(corte["traza"]), len(r["traza"]))
    bit = bool(np.array_equal(corte["traza"][:k, :6], r["traza"][:k, :6], equal_nan=True))
    hc = {int(i): v for i, v in corte["psnr"]}
    psnr_igual = all(abs(hc[int(i)] - v) == 0 for i, v in p if int(i) in hc)
    e = r.get("estado", {})
    return dict(n=n, fin=terminada(r) or cortada(r), estado=estado_txt(r),
                llego=terminada(r) and int(r["it"][-1]) >= 100_000, it=int(r["it"][-1]), best=b[1], it_best=int(b[0]),
                ultimo=p[-1, 1], T=float(r["T"][-1]), Tmin=float(np.min(r["T"])),
                lam=float(r["lam"][-1]), rech=int(rech.sum()), reb=int(r["reb"].sum()),
                maxracha=maxracha, cota_sola=sola, horas=e.get("horas"), e6=e6, bit=bit, k=k,
                psnr_igual=psnr_igual)


F = [fila(n) for n in FINALES]
prov = "" if all(f["fin"] for f in F) else " — **PROVISIONAL: hay corridas en vuelo**"
L = [f"# Números de las corridas finales{prov}", "",
     f"Checkpoint oficial: **{ORAC_PSNR:.3f} dB** (`runs/oracle-cbsd68.json`); paper 31.53.", "",
     "| | " + " | ".join(f["n"] for f in F) + " |", "|---|" + "---|" * len(F)]
for k, t, fmt in (("estado", "estado", str),
                  ("it", "iteraciones en la traza", "{:,}".format),
                  ("best", "mejor PSNR (dB)", "**{:.3f}**".format),
                  ("it_best", "en la it", "{:,}".format),
                  ("ultimo", "PSNR de la última evaluación", "{:.3f}".format),
                  ("T", "T al final", "{:.5f}".format), ("Tmin", "T mínima", "{:.5f}".format),
                  ("lam", "λ al final", "{:.5f}".format),
                  ("rech", "lotes rechazados", "{:,}".format), ("reb", "rebobinados", str),
                  ("maxracha", "racha máx. de rechazos", str),
                  ("cota_sola", "lotes agarrados SÓLO por la cota 10⁴", str),
                  ("horas", "horas", lambda v: "—" if v is None else f"{v:.1f}")):
    L.append(f"| {t} | " + " | ".join(fmt(f[k]) for f in F) + " |")

ok = lambda c: "✔ se cumple" if c else "✘ falla"  # noqa: E731
dif = abs(F[0]["best"] - F[1]["best"])
# E1 y E5 (y la 1 original) hablan de la it 100 000: sólo cuentan las corridas que llegaron.
L100 = [f for f in F if f["llego"]]
sin100 = ", ".join(f"{f['n'][-2:]}: {f['estado']}" for f in F if not f["llego"])


def a_100k(cond, fmt):
    medido = ", ".join(fmt(f) for f in L100) or "—"
    if not L100:
        return medido, "sin resultado: ninguna llegó a la 100k"
    v = ok(all(cond(f) for f in L100))
    return medido, v + (f" (sólo {', '.join(f['n'][-2:] for f in L100)}; {sin100})" if sin100 else "")
e6 = [x for f in F for x in f["e6"]]
P = [  # (#, predicción, medido, veredicto) — pre-registro docs/bitacora.md:3577-3601
     ("1", "T a la it 100 000 en [0.018, 0.032]", a_100k(lambda f: True, lambda f: f"{f['T']:.4f}")[0],
      "✘ reemplazada por E1 antes de terminar (los parciales ya la falsificaban)"),
     ("2", f"T nunca baja de la pared δ = {DELTA_ADAM:.5f}",
      ", ".join(f"mín {f['Tmin']:.4f}" for f in F),
      ok(all(f["Tmin"] >= DELTA_ADAM for f in F)) + " — NO INFORMATIVA (E2): no podía fallar"),
     ("3", "mejor PSNR ≥ 31.40 dB", ", ".join(f"{f['best']:.3f}" for f in F),
      ok(all(f["best"] >= 31.40 for f in F))),
     ("4", "ninguna se traba: racha de rechazos < 1 000", ", ".join(str(f["maxracha"]) for f in F),
      ok(all(f["maxracha"] < 1000 for f in F))),
     ("5", "las dos semillas a < 0.33 dB (mejor PSNR)", f"{dif:.3f} dB", ok(dif < 0.33)
      + ("" if all(f["llego"] for f in F) else f" — ojo: {sin100}, su mejor es de la it "
         + ", ".join(f"{f['it_best']:,}" for f in F if not f["llego"]))),
     ("6", "la cota 10⁴ nunca agarra un lote sola", ", ".join(str(f["cota_sola"]) for f in F),
      ok(all(f["cota_sola"] == 0 for f in F))),
     # enmienda del 2026-10-05, docs/bitacora.md:3687-3714
     ("E1", "T a la it 100 000 en [0.050, 0.075]",
      *a_100k(lambda f: 0.050 <= f["T"] <= 0.075, lambda f: f"{f['T']:.4f}")),
     ("E2", "ningún brazo llega a T < δ", "= la 2", "no informativa por diseño (lo dice la enmienda)"),
     ("E3", "s1 reproduce su trayectoria cortada hasta la it 18 000",
      f"traza idéntica en {F[1]['k']:,} filas: {F[1]['bit']}; PSNR idénticas: {F[1]['psnr_igual']}",
      ok(F[1]["bit"] and F[1]["psnr_igual"]) + " (bit a bit)"),
     ("E4", "s0 ídem", f"traza idéntica en {F[0]['k']:,} filas: {F[0]['bit']}; "
      f"PSNR idénticas: {F[0]['psnr_igual']}", ok(F[0]["bit"] and F[0]["psnr_igual"]) + " (bit a bit)"),
     ("E5", "PSNR a la it 100 000 en [31.45, 31.60]",
      *a_100k(lambda f: 31.45 <= f["ultimo"] <= 31.60, lambda f: f"{f['ultimo']:.3f}")),
     ("E6", "tras rebobinar, ‖g‖/mediana restaurada finita y en la banda de v1 [0.72, 1.04]",
      ", ".join(f"{x:.3f}" for x in e6) or "sin rebobinados",
      "—" if not e6 else (("✔ finitas" if all(np.isfinite(e6)) else "✘ no finitas")
                          + f"; {sum(0.72 <= x <= 1.04 for x in e6)} de {len(e6)} dentro de la banda"))]
L += ["", f"## Predicciones registradas antes de correr{prov}", "",
      "| # | predicción | medido (s0, s1) | veredicto |", "|---|---|---|---|"]
L += [f"| {a} | {b} | {c} | {d if not prov else d + ' (por ahora)'} |" for a, b, c, d in P]
L += ["", "1–6: pre-registro original. E1–E6: enmienda del 10-05, escrita antes de relanzar; E1 "
      "reemplaza a la 1 y las demás son nuevas (la 5 original sigue vigente). 1, E1 y E5 hablan de la it "
      "100 000 y se leen sólo sobre las corridas que llegaron (la s0 se cortó a mano en la 66k)."]
(SALIDA / "numeros.md").write_text("\n".join(L) + "\n")
json.dump(F, open(SALIDA / "numeros.json", "w"), indent=1, default=float)
print("\n".join(L))
