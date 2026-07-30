import io
import pandas as pd
import scanpy as sc
import gcsfs

fs = gcsfs.GCSFileSystem()

vcc_base_path = "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/"

train_adata_path = vcc_base_path + "train/adata_Training.h5ad"
val_adata_path = vcc_base_path + "validation/adata_Validation.h5ad"

print(f"Training data path: {train_adata_path}")
print(f"Validation data path: {val_adata_path}")

with fs.open(train_adata_path) as f:
    adata = sc.read_h5ad(f)
    
with fs.open(val_adata_path) as f:
    adata_val = sc.read_h5ad(f)

adata.write_h5ad("data/adata_Training.h5ad")
adata_val.write_h5ad("data/adata_Validation.h5ad")