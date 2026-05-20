import json

with open("playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i in range(max(0, len(nb['cells']) - 20), len(nb['cells'])):
    cell = nb['cells'][i]
    content = "".join(cell['source'])[:80].replace("\n", " ")
    print(f"Cell {i} ({cell['cell_type']}): {content}")
