import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package", "dev_notebooks", "fits_measure_2.ipynb"))

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source_str = "".join(cell["source"])
    if "Jy" in source_str or "Jy" in cell.get("outputs", []):
        print(f"Cell {i} has 'Jy':")
        print(source_str)
        print("="*40)
