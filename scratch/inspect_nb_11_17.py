import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(11, 18):
    cell = nb["cells"][idx]
    print(f"Cell {idx} ({cell['cell_type']}):")
    print("".join(cell.get("source", [])))
    print("-" * 80)
