# Genshin Lore Analyzer

This repository contains a small analysis pipeline for character lore text (cleaning, embeddings, sentiment, keywords, and similarity). It is organized as a set of Jupyter notebooks plus lightweight helper modules in `src/`.

**Repository layout**
- `notebooks/` : Jupyter notebooks for each step of the pipeline
	- `01_text_cleaning.ipynb` — text cleaning and preprocessing
	- `02_embeddings.ipynb` — generate or load embeddings
	- `03_sentiment.ipynb` — sentiment analysis, keyword extraction, save results, and compute similarity
	- `04_keywords.ipynb` — focused keyword extraction
	- `05_clustering.ipynb` — (optional) clustering experiments
- `src/` : small helper modules (`preprocessing.py`, `keywords.py`, `sentiment.py`, `clustering.py`)
- `data/`
	- `raw/` : original raw text files (not committed)
	- `processed/` : cleaned CSVs and processed outputs (some processed outputs may be committed)
- `requirements.txt` : Python dependencies
- `.gitignore` : Ignore environment and large data (includes `venv/`, `.vscode/`, and raw data folders)

**Quick summary**
- Purpose: Explore and analyze lore text (clean, extract keywords, compute sentiment, build embeddings, compute similarity)
- Target user: researcher or developer running notebooks locally with a Python virtualenv

**Setup (Windows / PowerShell)**
1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If using `TextBlob`, download corpora:

```powershell
python -m textblob.download_corpora
```

**Data expectations**
- Place raw text files in `data/raw/` (these are ignored by `.gitignore`). Example files used in this project:
	- `data/raw/raiden_lore.txt`
	- `data/raw/venti_lore.txt`
	- `data/raw/zhongli_lore.txt`
- Notebook `01_text_cleaning.ipynb` will write cleaned tabular data to `data/processed/cleaned.csv` (small processed outputs may be included in the repository). Do not commit large raw data or binary embeddings.

**How to run the notebooks**
Open the repository in VS Code or Jupyter Lab and run the notebooks in order. Recommended order:

1. `notebooks/01_text_cleaning.ipynb` — produce `data/processed/cleaned.csv`
2. `notebooks/02_embeddings.ipynb` — generate or load embeddings (writes `data/processed/embeddings.npy` if you generate them)
3. `notebooks/03_sentiment.ipynb` — computes sentiment and keywords, saves `data/processed/final_results.csv` and `data/processed/similarity_matrix.csv`
4. `notebooks/04_keywords.ipynb` — alternative keyword flow
5. `notebooks/05_clustering.ipynb` — optional clustering analyses

Notes when running:
- Cells contain loader checks that attempt to load `data/processed/cleaned.csv` into a DataFrame `df` if it's not already defined in the kernel.
- The notebooks include guards to handle missing embeddings: if `embeddings.npy` is missing, a numeric fallback is used to allow the similarity step to run (replace with real embeddings for production results).

**Generating real embeddings**
- The repo currently expects embeddings as a NumPy array saved with `np.save('data/processed/embeddings.npy', embeddings)` where `embeddings` is shape `(n_documents, embedding_dim)`.
- To generate embeddings with a transformer model (example using `sentence-transformers`):

```powershell
pip install sentence-transformers
python - <<'PY'
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

df = pd.read_csv('data/processed/cleaned.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['Cleaned_Text'].fillna('').astype(str).tolist()
emb = model.encode(texts, show_progress_bar=True)
np.save('data/processed/embeddings.npy', emb)
PY
```

After creating `embeddings.npy` rerun `notebooks/03_sentiment.ipynb` (embedding load + similarity cells) to compute a true similarity matrix.

**Git & repository policies**
- `.gitignore` was added to avoid committing virtual environments and raw data. Currently ignored entries include `venv/`, `.vscode/`, and `data/raw/`.
- Committed files should be code, notebooks, small processed outputs, and any scripts necessary to reproduce analysis.
- To push changes:

```powershell
git add <files>
git commit -m "Describe changes"
git push origin main
```

**Troubleshooting**
- If a notebook cell complains `df` is undefined, run the first loader cell in that notebook (Cell 1) to load `data/processed/cleaned.csv`.
- If you see errors loading embeddings, ensure `data/processed/embeddings.npy` contains a numeric NumPy array. The notebooks attempt to coerce and fallback, but you will get meaningful results only with real embeddings.
- On Windows, Git may warn about LF/CRLF conversions when touching files from different environments — these are warnings, not fatal errors.

**Contributors / Credits**
- Repo owner: `Bingeeverything` (GitHub)


