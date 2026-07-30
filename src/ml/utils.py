import os
import mygene
import pandas as pd
import numpy as np


def get_gene_embeddings(path):
    # check if cached embeddings exist
    cache_path = os.path.join(path, "gene_to_emb.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    emb = pd.read_csv(os.path.join(path, "ESM2/ESM2_emb.csv"), header=None)

    with open(os.path.join(path, "ESM2/ESM2_genelist.txt"), "r") as f:
        ids = [line.strip() for line in f]

    id_to_emb = {gene_id: emb.iloc[i].values for i, gene_id in enumerate(ids)}

    mg = mygene.MyGeneInfo()

    res = mg.querymany(ids, scopes="entrezgene", fields="symbol", species="human")

    gene_to_id = {item["symbol"]: item["query"] for item in res}

    gene_to_emb = {gene: id_to_emb[gene_to_id[gene]] for gene in gene_to_id}

    # cache the embeddings
    pd.to_pickle(gene_to_emb, cache_path)

    return gene_to_emb


def get_pseudobulks(adata, perturbation_genes):
    pert_genes = []
    pseudobulks = []
    for gene in perturbation_genes:
        gene_idx = adata.obs.target_gene == gene
        pert_genes.append(gene)
        pseudobulks.append(adata[gene_idx].X.toarray().mean(axis=0))

    return np.array(pert_genes), np.stack(pseudobulks)