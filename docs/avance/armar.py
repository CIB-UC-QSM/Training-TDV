"""Embebe las figuras como data URI en la plantilla y escribe avance.html."""
import base64, os, re
D = "docs/avance"
html = open(f"{D}/plantilla.html").read()
for k in re.findall(r"\{\{(f\d)\}\}", html):
    p = next((f"{D}/salida/{n}" for n in os.listdir(f"{D}/salida") if n.startswith(k + "_")), None)
    if p:
        html = html.replace("{{%s}}" % k, "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode())
    else:
        html = re.sub(r'<figure><img src="\{\{%s\}\}".*?</figure>' % k,
                      '<p class="note">Figura en cálculo: se agrega al terminar.</p>', html, flags=re.S)
open(f"{D}/avance.html", "w").write(html); print(len(html) // 1024, "kB")
