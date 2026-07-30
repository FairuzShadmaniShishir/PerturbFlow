# Virtual Cell
A virtual cell is a program that answers the question: "What happens to a cell under a
given perturbation?" Here, "what happens" means a change in gene expression, and a
"perturbation" is a gene knockdown. The model takes an embedding of the perturbed
gene, encodes the control cell's expression state, and predicts the resulting
perturbed expression profile.

⚠️ This is an educational project. It illustrates the main steps of building a virtual
cell - it is not a competitive perturbation-response model. The point is to understand
the ideas and have a base to play with and improve.

## How it works

The model has two parts:

```
control expression ──► scGPT-style gene encoder ──► cell-state embedding ──┐
                                                                            ├─► flow-matching
                                       perturbation (ESM2) embedding ───────┘   velocity field
                                                                                     │
                                                                                     ▼
                                                          integrate ODE from t=0 (control)
                                                          to t=1  ──►  predicted expression
```

- **Gene encoder** - genes are grouped into fixed-size chunks (to keep attention
  affordable at thousands of genes); each chunk gets a value embedding plus a learned
  identity embedding, and a small Transformer lets chunks attend to each other. A CLS
  token pools this into a single cell-state embedding - more robust to per-gene
  noise/dropout than feeding a raw expression vector straight into an MLP.
- **Flow matching** - instead of predicting one static "delta" and adding it to the
  control profile, the model learns a time-conditioned velocity field over the
  straight-line path between control and perturbed expression (conditional flow
  matching). At inference, the predicted expression is produced by numerically
  integrating this learned ODE from the control state, step by step, rather than a
  single fixed jump.

Key ideas covered in the tutorial and implemented here:

* Single-cell RNA-seq perturbation data
* Representing perturbations with ESM2 gene embeddings
* Splitting data by perturbations
* Restricting to highly variable genes and using pseudobulk expression for a stable signal
* Chunked gene-attention encoding of the control cell state
* Predicting expression change via conditional flow matching (ODE integration) instead
  of a single static delta
* Comparing against a simple baseline

## Project structure

```
src/
  ml/
    VirtualCell.py          # the model (chunked gene encoder + flow-matching field)
    PerturbationDataset.py  # PyTorch dataset
    utils.py                # gene embeddings + pseudobulk helpers
    eval.py                 # loss, baseline metrics, Pearson delta correlation
  scripts/
    download_data.py        # fetch training/validation data from GCS
    train.py                 # training loop (saves checkpoints to models/)
    precompute_artifacts.py  # shrinks data for deployment (see Deployment below)
    app.py                    # Streamlit app to explore predictions
  notebooks/
    playground.ipynb        # exploration
data/                       # data + embeddings (gitignored)
artifacts/                  # small precomputed files used by the deployed app
models/                     # saved checkpoints (gitignored)
```

## Setup

Requires Python 3.11+.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes the `ml` package importable
```

## Usage

All commands are run from the repository root.

### 1. Download the data

```
python src/scripts/download_data.py
```

This pulls the Virtual Cell Challenge training and validation `.h5ad` files into
`data/`. You also need the ESM2 gene embeddings in `data/ESM2/` (`ESM2_emb.csv` and
`ESM2_genelist.txt`) - see the resources below.

### 2. Train the model

```
python src/scripts/train.py
```

Checkpoints (model config + weights + the list of highly variable genes) are written
to `models/` as `model_<timestamp>.pth`, keeping the best validation loss.

### 3. Explore predictions

```
streamlit run src/scripts/app.py
```

Pick a perturbation gene and see the top predicted expression changes, plus an
adjustable slider for how many ODE integration steps the flow-matching model takes at
inference (more steps = smoother/more accurate prediction, at the cost of speed).

Note: `app.py` loads a specific checkpoint filename. After training your own model,
update the `CHECKPOINT_PATH` in `src/scripts/app.py` to point at your checkpoint.

## Deploying the app (Streamlit Community Cloud)

The full training `.h5ad` is too large to ship to a public repo or a hosted app, and
isn't needed at inference time anyway. Before deploying, precompute the small pieces
the app actually needs:

```
python src/scripts/precompute_artifacts.py --checkpoint models/model_<RUN>.pth
```

This writes `artifacts/ctrl_mean.npy` and `artifacts/gene_names.json`. Commit
`artifacts/` to the repo; keep `data/*.h5ad` out of it (already gitignored).

Then:

1. Push the repo to GitHub (public, unless you're on a paid Streamlit plan). If
   `models/model_<RUN>.pth` is over GitHub's 100 MB limit, track it with
   [Git LFS](https://git-lfs.com/) first.
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   **New app**, and point it at this repo with `src/scripts/app.py` as the main file.
3. Deploy. The first build installs dependencies (a couple of minutes); later commits
   redeploy automatically.

Streamlit Community Cloud apps get roughly 1 CPU / 1 GB RAM - comfortable for this
model's size on CPU, but worth checking locally if you scale up `hidden_dim`,
`emb_dim`, or the number of gene chunks substantially.
