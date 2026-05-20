import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Mostrar celda 1
print("Cell 1 (Imports):")
print("".join(nb["cells"][1]["source"]))
print("=" * 80)

# Mostrar las celdas 17, 18, 19
print("Cell 17 (Visualización original):")
print("".join(nb["cells"][17]["source"])[:300])
print("-" * 50)
print("Cell 18 (Markdown de confianza):")
print("".join(nb["cells"][18]["source"]))
print("-" * 50)
print("Cell 19 (Código de confianza):")
print("".join(nb["cells"][19]["source"])[:800])
print("-" * 50)
