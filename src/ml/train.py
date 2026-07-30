import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from ml.utils import get_gene_embeddings, get_pseudobulks
from ml.eval import compute_baseline_metrics, loss_fn, pearson_diff_corr
from ml.PerturbationDataset import PerturbationDataset
from ml.VirtualCell import VirtualCell
import os

RUN = datetime.now().strftime("%Y%m%d_%H%M%S")
RANDOM_SEED = 23
TRAIN_VAL_SPLIT = 0.8
HIDDEN_DIM = 16
MAX_EPOCHS = 100
PATIENCE = 8
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load data
print("Loading data...")
data = sc.read_h5ad("data/adata_Training.h5ad")
sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data)

# Load gene embeddings
print("Loading gene embeddings...")
gene_to_emb = get_gene_embeddings("data")

# Split genes into train and val
genes = sorted(set(data.obs.target_gene) - {"non-targeting"})
print(f"Total genes: {len(genes)}")

# Filter genes with embeddings
genes = [gene for gene in genes if gene in gene_to_emb]
print(f"Genes with embeddings: {len(genes)}")

train_genes = np.random.choice(list(genes), size=int(len(genes) * TRAIN_VAL_SPLIT), replace=False)
print(f"Train genes: {len(train_genes)}")
val_genes = sorted(set(genes) - set(train_genes))
print(f"Val genes: {len(val_genes)}")

# Sample split
ctrl_idx = data.obs.target_gene == "non-targeting"
train_idx = data.obs.target_gene.isin(train_genes)
val_idx = data.obs.target_gene.isin(val_genes)

train_data = data[train_idx | ctrl_idx].copy()
sc.pp.highly_variable_genes(train_data, n_top_genes=2000)
highly_variable_genes = list(train_data.var_names[train_data.var.highly_variable])
data = data[:, highly_variable_genes]

X_ctrl = data[ctrl_idx].X.toarray()
mean_ctrl = X_ctrl.mean(axis=0, keepdims=True)
pert_train, y_train = get_pseudobulks(data, train_genes)
pert_val, y_val = get_pseudobulks(data, val_genes)
del data

train_dataset = PerturbationDataset(mean_ctrl, pert_train, y_train, gene_to_emb)
val_dataset = PerturbationDataset(mean_ctrl, pert_val, y_val, gene_to_emb)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model
model_config = {
    "expr_dim": X_ctrl.shape[1],
    "emb_dim": len(next(iter(gene_to_emb.values()))),
    "hidden_dim": HIDDEN_DIM,
}
model = VirtualCell(**model_config).to(device)
mean_expr = torch.tensor(y_train.mean(axis=0), dtype=torch.float32).to(device)
mean_ctrl = torch.tensor(mean_ctrl, dtype=torch.float32).to(device)
with torch.no_grad():
    #model.delta_net[-1].bias.copy_((mean_expr - mean_ctrl).squeeze(0))
    model.velocity_field.net[-1].bias.copy_((mean_expr - mean_ctrl).squeeze(0))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# Baseline
baseline_loss, baseline_corr = compute_baseline_metrics(val_loader, mean_expr, device)
print(f"Baseline Loss: {baseline_loss:.4f} - Baseline Pseudobulk Pearson: {baseline_corr:.4f}")

# Epoch loop
best_val_loss = float("inf")
patience_counter = 0
for epoch in range(MAX_EPOCHS):
    #  Train loop
    train_loss = 0.0
    model.train()
    for batch_ctrl_expr, batch_pert_emb, batch_target_expr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        optimizer.zero_grad()

        batch_ctrl_expr = batch_ctrl_expr.to(device)
        batch_pert_emb = batch_pert_emb.to(device)
        batch_target_expr = batch_target_expr.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_expr = model(batch_ctrl_expr, batch_pert_emb)
            loss = loss_fn(pred_expr, batch_target_expr, batch_ctrl_expr)
        # pred_expr = model(batch_ctrl_expr, batch_pert_emb)
        #
        # loss = loss_fn(pred_expr, batch_target_expr, batch_ctrl_expr)

        train_loss += loss.item() * batch_ctrl_expr.size(0)

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    #  Val loop
    val_loss = 0.0
    val_corr = 0.0
    model.eval()
    with torch.no_grad():
        for batch_ctrl_expr, batch_pert_emb, batch_target_expr in val_loader:
            batch_ctrl_expr = batch_ctrl_expr.to(device)
            batch_pert_emb = batch_pert_emb.to(device)
            batch_target_expr = batch_target_expr.to(device)

            pred_expr = model(batch_ctrl_expr, batch_pert_emb)

            loss = loss_fn(pred_expr, batch_target_expr, batch_ctrl_expr)

            val_loss += loss.item() * batch_ctrl_expr.size(0)
            val_corr += pearson_diff_corr(pred_expr, batch_target_expr, batch_ctrl_expr) * batch_ctrl_expr.size(0)

    val_loss /= len(val_loader.dataset)
    val_corr /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Corr: {val_corr:.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # save model checkpoint (both config and weights)
        torch.save({
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "highly_variable_genes": highly_variable_genes
        }, f"models/model_{RUN}.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break