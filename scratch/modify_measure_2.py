import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package", "dev_notebooks", "fits_measure_2.ipynb"))

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

modified = False
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("plt.xlabel('Pixel Amplitude (Jy)')" in line for line in cell["source"]):
        cell["source"] = [
            "myHist100(img2)\n",
            "plt.yscale('log')\n",
            "plt.title('Distribution of Pixel Amplitudes')\n",
            "bunit = hdr.get('BUNIT', 'Counts')\n",
            "plt.xlabel(f'Pixel Amplitude ({bunit})')\n",
            "plt.ylabel('Number of Pixels')\n",
            "plt.show()"
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
        modified = True
        print(f"Updated Cell {i} in fits_measure_2.ipynb")

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Successfully saved fits_measure_2.ipynb.")
else:
    print("Warning: Cell 4 not found or already modified.")
