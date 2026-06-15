import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package", "dev_notebooks", "fits_measure_2.ipynb"))

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(6, len(nb["cells"])):
    cell = nb["cells"][i]
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        print(f"Cell {i} (Code):")
        print(source_str)
        print("-" * 50)
