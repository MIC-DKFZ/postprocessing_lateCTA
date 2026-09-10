# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

outfile = "/media/E132-Projekte/Projects/2025_MartinezMora_TimeAwareVODetection/models_stroke_amsterdamproject/results_development_table.html"

# --------------------------
# Data
# --------------------------
data = {
    "Method": [
        "Baseline",
        "False occlusion removal",
        "Time vessel map (extra channel)",
    ],
    "FROC": [(0.45, "0.40–0.50"), (0.47, "0.42–0.52"), (0.40, "0.35–0.44")],
    "Occlusion Sensitivity": [
        (0.63, "0.56–0.69"),
        (0.63, "0.56–0.69"),
        (0.94, "0.91–0.97"),
    ],
    "False occlusions / scan": [
        (2.13, "1.94–2.34"),
        (1.72, "1.56–1.88"),
        (83.47, "80.19–86.41"),
    ],
}

df = pd.DataFrame(data)


# --------------------------
# Function to map value to color (column-wise)
# --------------------------
def column_gradient(val, col_values, reverse=False):
    """
    val : numeric value
    col_values : list of numeric values in the column
    reverse : True if lower is better
    """
    vmin, vmax = min(col_values), max(col_values)
    norm = (val - vmin) / (vmax - vmin)
    norm = max(0, min(1, norm))  # clip
    if reverse:
        norm = 1 - norm
    cmap = plt.get_cmap("RdYlGn")  # red->yellow->green
    rgba = cmap(norm)
    return matplotlib.colors.to_hex(rgba)


# --------------------------
# Precompute column values for gradients
# --------------------------
col_values_dict = {}
for col in df.columns[1:]:
    col_values_dict[col] = [v[0] for v in df[col]]

# --------------------------
# Build HTML table with column-wise gradient
# --------------------------
html = '<table border="1" cellpadding="6" style="border-collapse: collapse; font-family: sans-serif;">'

# Header
html += "<tr>"
for col in df.columns:
    html += f"<th>{col}</th>"
html += "</tr>"

# Rows
for _, row in df.iterrows():
    html += "<tr>"
    for col in df.columns:
        if col == "Method":
            html += f"<td>{row[col]}</td>"
        else:
            val, ci = row[col]
            reverse = col == "False occlusions / scan"  # lower is better
            color = column_gradient(val, col_values_dict[col], reverse=reverse)
            html += f'<td style="background-color:{color}">{val:.2f} [{ci}]</td>'
    html += "</tr>"

html += "</table>"

# Save to file
with open(outfile, "w") as f:
    f.write(html)
