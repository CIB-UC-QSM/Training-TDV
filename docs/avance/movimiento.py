"""Cuánto se mueve cada tensor desde el init (semilla 0): ||W-W0||/||W0||."""
import sys, torch, json
sys.path.insert(0, ".")
from evaluate import load_vnet
from ddr.model import VNet2D
R = "runs"
cfg = torch.load(f"{R}/precision/rapido-seed0/best.pth", map_location="cpu", weights_only=False)["config"]
torch.manual_seed(0); w0 = VNet2D(cfg).state_dict()
res = {}
for tag, p in [("actual_4k", f"{R}/precision/rapido-seed0/best.pth"),
               ("v4_68k", f"{R}/replica-v4/checkpoint.pth"),
               ("ref100k_38k", f"{R}/referencia-100k/best.pth"),
               ("oficial", "checkpoints/Oficiales/tdv3-3-25-f32-color.pth")]:
    sd = load_vnet(p)[0].state_dict()
    res[tag] = {k: ((sd[k] - w0[k]).norm() / w0[k].norm()).item() for k in w0
                if k.endswith("weight")}
keys = list(res["v4_68k"])
print(f"{'tensor':40s}" + "".join(f"{t:>13s}" for t in res))
for k in keys:
    print(f"{k:40s}" + "".join(f"{res[t][k]:13.3f}" for t in res))
json.dump(res, open("docs/avance/salida/movimiento.json", "w"))
