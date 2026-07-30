import numpy as np
import pandas as pd
import scanpy as sc
import streamlit as st
import torch

from ml.utils import get_gene_embeddings
from ml.VirtualCell import VirtualCell

CHECKPOINT_PATH = "models/model_20260729_165856.pth"
DATA_PATH = "data/adata_Training.h5ad"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint["model_config"]
    highly_variable_genes = checkpoint["highly_variable_genes"]

    model = VirtualCell(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # eval mode -> forward() runs the full multi-step ODE sample

    return model, highly_variable_genes, model_config


@st.cache_resource
def load_data(_highly_variable_genes):
    data = sc.read_h5ad(DATA_PATH)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    data = data[:, _highly_variable_genes]

    X_ctrl = data[data.obs.target_gene == "non-targeting"].X.toarray()
    gene_names = data.var_names.tolist()

    return X_ctrl, gene_names


@st.cache_resource
def load_gene_embeddings():
    return get_gene_embeddings("data")


def predict(model, ctrl_mean, gene_to_emb, gene, n_ode_steps):
    ctrl = torch.tensor(ctrl_mean, dtype=torch.float32).unsqueeze(0).to(device)
    emb = torch.tensor(gene_to_emb[gene], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.sample(ctrl, emb, n_steps=n_ode_steps)

    return pred.squeeze(0).cpu().numpy()


st.set_page_config(page_title="Virtual Cell", layout="wide")
st.title("Virtual Cell")
st.caption(f"Running on device: **{device}**")

model, highly_variable_genes, model_config = load_model()
X_ctrl, gene_names = load_data(highly_variable_genes)
gene_to_emb = load_gene_embeddings()
ctrl_mean = X_ctrl.mean(axis=0)

col1, col2 = st.columns([2, 1])
with col1:
    gene = st.selectbox("Perturbation gene", sorted(gene_to_emb.keys()))
with col2:
    n_ode_steps = st.slider(
        "ODE integration steps",
        min_value=1, max_value=50,
        value=model_config.get("n_ode_steps", 10),
        help="More steps = smoother/more accurate flow-matching integration, at the cost of more forward passes.",
    )

with st.spinner("Predicting perturbation effect..."):
    pred_mean = predict(model, ctrl_mean, gene_to_emb, gene, n_ode_steps)

delta = pred_mean - ctrl_mean

df = pd.DataFrame({
    "gene": gene_names,
    "control": ctrl_mean,
    "predicted": pred_mean,
    "delta": delta,
    "abs_delta": np.abs(delta),
})

n_top = st.slider("Number of top genes to show", min_value=10, max_value=200, value=50, step=10)
top_df = df.sort_values("abs_delta", ascending=False).head(n_top).reset_index(drop=True)

tab_table, tab_chart, tab_updown = st.tabs(["Table", "Chart", "Top up / down"])

with tab_table:
    st.dataframe(
        top_df.drop(columns="abs_delta"),
        use_container_width=True,
        hide_index=True,
    )

with tab_chart:
    chart_df = top_df.set_index("gene")[["delta"]].sort_values("delta")
    st.bar_chart(chart_df, use_container_width=True)

with tab_updown:
    up_col, down_col = st.columns(2)
    with up_col:
        st.subheader("Top upregulated")
        st.dataframe(
            df.sort_values("delta", ascending=False).head(20)[["gene", "control", "predicted", "delta"]],
            use_container_width=True,
            hide_index=True,
        )
    with down_col:
        st.subheader("Top downregulated")
        st.dataframe(
            df.sort_values("delta", ascending=True).head(20)[["gene", "control", "predicted", "delta"]],
            use_container_width=True,
            hide_index=True,
        )

st.download_button(
    "Download full prediction as CSV",
    data=df.drop(columns="abs_delta").to_csv(index=False),
    file_name=f"virtualcell_prediction_{gene}.csv",
    mime="text/csv",
)