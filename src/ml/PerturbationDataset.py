import torch

from torch.utils.data import Dataset


class PerturbationDataset(Dataset):
    def __init__(self, ctrl_expr, pert_genes, target_expr, gene_to_emb):
        self.ctrl_expr = torch.tensor(ctrl_expr, dtype=torch.float32)
        self.pert_genes = pert_genes
        self.target_expr = torch.tensor(target_expr, dtype=torch.float32)
        self.gene_to_emb = gene_to_emb

    def __len__(self):
        return len(self.pert_genes)

    def __getitem__(self, idx):
        pert_gene = self.pert_genes[idx]
        pert_emb = torch.tensor(self.gene_to_emb[pert_gene], dtype=torch.float32)

        target_expr = self.target_expr[idx]

        ctrl_idx = torch.randint(0, self.ctrl_expr.shape[0], (1,)).item()
        ctrl_expr = self.ctrl_expr[ctrl_idx]

        return ctrl_expr, pert_emb, target_expr