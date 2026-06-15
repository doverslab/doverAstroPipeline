import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package", "dev_notebooks", "fits_cal_and_stack.ipynb"))

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and update cells
modified_markdown = False
modified_plot_cell = False
modified_best_cell = False

for i, cell in enumerate(nb["cells"]):
    # 1. Update Markdown Cell
    if cell["cell_type"] == "markdown" and any("## 6. Generate Comparison Plots" in line for line in cell["source"]):
        cell["source"] = [
            "## 6. Generate Comparison Plot\n",
            "\n",
            "We create a bar chart grouped by star comparing:\n",
            "- Wavelet vs. Polynomial 7D background subtraction SNR"
        ]
        modified_markdown = True
        print(f"Updated Markdown Cell at index {i}")
        
    # 2. Update Plotting Code Cell
    elif cell["cell_type"] == "code" and any("col_a = [" in line and "Run_A_Post_Cal_Wavelet" in line for line in cell["source"]):
        cell["source"] = [
            "star_ids = snr_df[\"Star_ID\"]\n",
            "x = np.arange(len(star_ids))\n",
            "width = 0.35\n",
            "\n",
            "# Find actual columns (accounts for filename suffixes in stack_images output)\n",
            "col_a = [c for c in snr_df.columns if \"Run_A_Wavelet\" in c][0]\n",
            "col_b = [c for c in snr_df.columns if \"Run_B_Poly\" in c][0]\n",
            "\n",
            "# Plot: Wavelet vs Polynomial background subtraction\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.bar(x - width/2, snr_df[col_a], width, label=\"Wavelet bg (Run A)\", color=\"teal\")\n",
            "plt.bar(x + width/2, snr_df[col_b], width, label=\"Polynomial 7D bg (Run B)\", color=\"purple\")\n",
            "avg_line = (snr_df[col_a] + snr_df[col_b]) / 2\n",
            "plt.plot(x, avg_line, color=\"red\", marker=\"o\", label=\"Average SNR\", linewidth=2)\n",
            "plt.xticks(x, star_ids)\n",
            "plt.xlabel(\"Celestial Objects\")\n",
            "plt.ylabel(\"SNR\")\n",
            "plt.title(\"Wavelet vs Polynomial 7D Background Subtraction SNR Comparison\")\n",
            "plt.legend()\n",
            "plt.savefig(\"plot_wavelet_vs_poly.png\", bbox_inches='tight')\n",
            "plt.show()"
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
        modified_plot_cell = True
        print(f"Updated Plotting Code Cell at index {i}")
        
    # 3. Update Save Best Stacked Image Cell
    elif cell["cell_type"] == "code" and any("col_d = [" in line and "Run_D_Post_Cal_No_Clip" in line for line in cell["source"]):
        cell["source"] = [
            "avg_snrs = {\n",
            "    \"Run_A_Wavelet\": snr_df[col_a].mean(),\n",
            "    \"Run_B_Poly\": snr_df[col_b].mean()\n",
            "}\n",
            "best_run = max(avg_snrs, key=avg_snrs.get)\n",
            "print(f\"Best Stacked Image is from: {best_run} with average SNR: {avg_snrs[best_run]:.4f}\")\n",
            "\n",
            "best_stacked_source = run_results[best_run]\n",
            "best_stacked_dest = \"best_stacked_image.fits\"\n",
            "\n",
            "hdul = fits.open(best_stacked_source)\n",
            "best_data = hdul[0].data\n",
            "best_header = hdul[0].header.copy()\n",
            "hdul.close()\n",
            "\n",
            "# Add descriptive header values\n",
            "best_header[\"STACKMETHOD\"] = (\"median\", \"Image stacking method\")\n",
            "best_header[\"SIGMACLIP\"] = (runs[best_run][\"sigma_clip\"], \"Sigma clipping threshold applied\")\n",
            "best_header[\"BGSUB\"] = (runs[best_run][\"bg_sub_method\"], \"Background subtraction method\")\n",
            "best_header[\"CALFRAME\"] = (runs[best_run][\"cal_frames_flux\"], \"Calibration on individual frames\")\n",
            "best_header[\"CALSTACK\"] = (runs[best_run][\"cal_stacked_flux\"], \"Calibration on final stacked image\")\n",
            "best_header[\"BESTRUN\"] = (best_run, \"Run name identifier\")\n",
            "\n",
            "avg_snr_val = avg_snrs[best_run]\n",
            "if np.isnan(avg_snr_val):\n",
            "    avg_snr_val = 0.0\n",
            "best_header[\"AVG_SNR\"] = (float(avg_snr_val), \"Average SNR of selected catalog stars\")\n",
            "\n",
            "best_hdu = fits.PrimaryHDU(data=best_data, header=best_header)\n",
            "best_hdul = fits.HDUList([best_hdu])\n",
            "best_hdul.writeto(best_stacked_dest, overwrite=True, output_verify=\"ignore\")\n",
            "best_hdul.close()\n",
            "print(f\"Saved best stacked image to: {best_stacked_dest}\")"
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
        modified_best_cell = True
        print(f"Updated Best Stack Code Cell at index {i}")

if not modified_markdown or not modified_plot_cell or not modified_best_cell:
    print(f"Warning: markdown={modified_markdown}, plot={modified_plot_cell}, best={modified_best_cell}")
else:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Successfully updated notebook and saved.")
