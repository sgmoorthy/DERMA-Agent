import nbformat

with open("notebooks/demo_discovery.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)

# We want to add visualisations.
md_cell1 = nbformat.v4.new_markdown_cell("## Perception (The Eyes): Saliency Heatmap")
code_cell1 = nbformat.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
from tools.pathology_utils import generate_saliency_heatmap

# Create a mock image
image = np.ones((256, 256, 3), dtype=np.uint8) * 200
image[100:150, 100:150] = [100, 100, 200]

heatmap = generate_saliency_heatmap(image)

plt.figure(figsize=(6, 6))
plt.imshow(heatmap)
plt.title("Agent Attention Heatmap")
plt.axis('off')
plt.show()""")

md_cell2 = nbformat.v4.new_markdown_cell("## Validation (The Hands): Survival Analysis")
code_cell2 = nbformat.v4.new_code_cell("""import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Mock data based on agent's execution
df = pd.DataFrame({
    "time": [10, 20, 30, 40, 50, 60, 70, 80],
    "event": [1, 0, 1, 1, 0, 1, 0, 1],
    "group": ["High", "Low", "High", "Low", "High", "Low", "High", "Low"]
})

fig, ax = plt.subplots(figsize=(8, 5))
kmf = KaplanMeierFitter()

for name, grouped_df in df.groupby("group"):
    kmf.fit(grouped_df["time"], grouped_df["event"], label=name)
    kmf.plot_survival_function(ax=ax)
    
plt.title("Kaplan-Meier Survival Curve (Discovered Feature)")
plt.ylabel("Survival Probability")
plt.xlabel("Timeline (Days)")
plt.show()""")

nb.cells.extend([md_cell1, code_cell1, md_cell2, code_cell2])

with open("notebooks/demo_discovery.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Notebook updated.")
