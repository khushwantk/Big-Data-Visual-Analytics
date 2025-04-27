Dataset has to be placed in the same directory as dashboard.py

Install the required packages :
```bash
pip install streamlit pandas numpy plotly scikit-learn umap-learn hdbscan networkx statsmodels
```

Alternatively
```
pip install -r req.txt
```

To run the visualization:
```bash
streamlit run dashboard.py
```



If you face any compatibility issues with package versions with hdbscan, it is recommended to use Conda or the UV Python package manager for better environment management and dependency resolution.

Install uv package manager (An extremely fast Python package)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Using UV:
```bash
uv venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
uv pip install -r req.txt
```

To run the visualization:
```bash
streamlit run dashboard.py
```

Use Conda (strongly recommended on macOS M1/ARM)
If you’re on an M1 Mac, switch to conda-forge where pre-built ARM binaries exist:

```bash
conda install -c conda-forge numpy scipy hdbscan
```
This ensures matching ABI versions for NumPy, SciPy and HDBSCAN without any compilation headaches.
