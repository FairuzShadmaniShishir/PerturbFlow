# import torch
# import torch.nn as nn
#
#
# class VirtualCell(nn.Module):
#     def __init__(self, expr_dim, emb_dim, hidden_dim):
#         super(VirtualCell, self).__init__()
#         self.delta_net = nn.Sequential(
#             nn.Linear(emb_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, expr_dim),
#         )
#         nn.init.zeros_(self.delta_net[-1].weight)
#         nn.init.zeros_(self.delta_net[-1].bias)
#
#     def forward(self, ctrl_expr, pert_emb):
#         pred_delta = self.delta_net(pert_emb)
#
#         return ctrl_expr + pred_delta


"""
Flow-matching VirtualCell model.

Upgrades over the original static-delta MLP:

  1. Gene-context state encoder (scGPT-style)
     Instead of feeding raw control expression straight into an MLP, each
     gene is tokenized as (gene-identity embedding + continuous value
     embedding) and passed through a small Transformer encoder with a
     learned [CLS]-style pooling token. Genes attend to each other, so the
     resulting cell-state embedding is context-aware and far less brittle
     to dropout/noise in single-cell counts than a linear projection of
     the raw vector.

  2. Conditional flow matching instead of a single static delta
     The model learns a time-conditioned velocity field
         v_theta(x_t, t, cond)
     over the straight-line path x_t = (1-t) * x0 + t * x1, where x0 is the
     control expression and x1 is the perturbed expression (Lipman et al.
     2023 / Tong et al. 2023, conditional flow matching). At inference the
     perturbed profile is produced by numerically integrating the ODE
     dx/dt = v_theta(x_t, t, cond) from t=0 to t=1 (multiple small steps),
     rather than one single "add a fixed delta" step. This gives a
     generative, multi-step model that can express more complex,
     non-additive transcriptional responses and is more robust to
     large/nonlinear perturbation effects.

The public API (`forward(ctrl_expr, pert_emb) -> predicted expression`)
is kept as a drop-in replacement for the original `VirtualCell`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous embedding for the flow-matching time variable t in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "time_dim must be even"
        self.dim = dim

    def forward(self, t):
        # t: (batch,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


import math

class GeneExpressionEncoder(nn.Module):
    """
    Chunked, scGPT-inspired cell-state encoder.

    With thousands of genes, a per-gene token is too expensive for full
    self-attention (cost scales with seq_len^2). Instead, genes are grouped
    into fixed-size chunks; each chunk's raw expression values are projected
    into a single token, combined with a learned per-chunk identity
    embedding. A Transformer encoder then lets chunks attend to each other,
    and a CLS token pools this into a cell-state embedding.

    e.g. expr_dim=2000, chunk_size=20 -> 100 chunk tokens + 1 CLS = 101
    tokens, instead of 2001 -- a ~400x reduction in attention cost.
    """

    def __init__(self, expr_dim, emb_dim, n_heads=4, n_layers=2, dropout=0.1, chunk_size=20):
        super().__init__()
        self.expr_dim = expr_dim
        self.emb_dim = emb_dim
        self.chunk_size = chunk_size
        self.n_chunks = math.ceil(expr_dim / chunk_size)
        self.padded_dim = self.n_chunks * chunk_size
        self.pad_len = self.padded_dim - expr_dim

        # learned identity embedding per chunk (analogous to gene_id_emb,
        # but one per group of genes rather than per individual gene)
        self.chunk_id_emb = nn.Embedding(self.n_chunks, emb_dim)
        # projects a chunk's raw expression values (chunk_size numbers) into emb_dim
        self.chunk_value_proj = nn.Sequential(
            nn.Linear(chunk_size, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(emb_dim)

        self.register_buffer("chunk_ids", torch.arange(self.n_chunks), persistent=False)

    def forward(self, expr):
        """
        expr: (batch, expr_dim)
        returns:
            cell_emb:     (batch, emb_dim)
            chunk_tokens: (batch, n_chunks, emb_dim) -- per-chunk context
                          (coarser than per-gene, since genes are grouped)
        """
        batch = expr.shape[0]

        if self.pad_len > 0:
            expr = F.pad(expr, (0, self.pad_len))  # zero-pad trailing genes

        chunks = expr.view(batch, self.n_chunks, self.chunk_size)  # (B, n_chunks, chunk_size)

        chunk_ids = self.chunk_ids.unsqueeze(0).expand(batch, -1)
        tokens = self.chunk_id_emb(chunk_ids) + self.chunk_value_proj(chunks)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.out_norm(self.encoder(tokens))

        return tokens[:, 0], tokens[:, 1:]


class VelocityField(nn.Module):
    """Predicts v_theta(x_t, t, cond) for the conditional flow-matching ODE."""

    def __init__(self, expr_dim, cond_dim, hidden_dim, time_dim=128, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        in_dim = expr_dim + cond_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expr_dim),
        )

        # Zero-init the final layer, echoing the original model's trick:
        # at initialization the field predicts zero velocity, so the ODE
        # solve starts out as the identity map (predicted = control).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t, t, cond):
        t_emb = self.time_mlp(self.time_emb(t))
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)


class VirtualCell(nn.Module):
    """
    Flow-matching virtual cell model. Drop-in replacement:
        forward(ctrl_expr, pert_emb) -> predicted expression, shape (B, expr_dim)

    For training, prefer calling `compute_flow_matching_loss` directly with
    paired (control, perturbed) expression, since that is the actual CFM
    objective this model is optimized for.
    """

    def __init__(
        self,
        expr_dim,
        emb_dim,
        hidden_dim,
        n_heads=4,
        n_layers=2,
        time_dim=128,
        dropout=0.1,
        n_ode_steps=10,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.n_ode_steps = n_ode_steps

        self.state_encoder = GeneExpressionEncoder(
            expr_dim=expr_dim,
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.pert_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        cond_dim = emb_dim * 2  # cell-state embedding + perturbation embedding
        self.velocity_field = VelocityField(
            expr_dim=expr_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            dropout=dropout,
        )

    def _condition(self, ctrl_expr, pert_emb):
        cell_emb, _ = self.state_encoder(ctrl_expr)
        pert_emb = self.pert_proj(pert_emb)
        return torch.cat([cell_emb, pert_emb], dim=-1)

    def compute_flow_matching_loss(self, ctrl_expr, pert_expr, pert_emb):
        """
        ctrl_expr: (B, expr_dim) control/source expression (x0)
        pert_expr: (B, expr_dim) observed perturbed/target expression (x1)
        pert_emb:  (B, emb_dim)  perturbation embedding
        """
        cond = self._condition(ctrl_expr, pert_emb)

        b = ctrl_expr.shape[0]
        t = torch.rand(b, device=ctrl_expr.device)
        t_ = t.unsqueeze(-1)

        x_t = (1 - t_) * ctrl_expr + t_ * pert_expr
        target_v = pert_expr - ctrl_expr  # straight-line conditional velocity

        pred_v = self.velocity_field(x_t, t, cond)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, ctrl_expr, pert_emb, n_steps=None):
        """Integrate dx/dt = v_theta(x_t, t, cond) from t=0 to t=1 via Euler steps."""
        n_steps = n_steps or self.n_ode_steps
        cond = self._condition(ctrl_expr, pert_emb)

        x = ctrl_expr
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + self.velocity_field(x, t, cond) * dt
        return x

    def forward(self, ctrl_expr, pert_emb):
        if self.training:
            # cheap single-Euler-step estimate, useful only as a logging
            # signal during training; the real objective is the CFM loss
            cond = self._condition(ctrl_expr, pert_emb)
            t0 = torch.zeros(ctrl_expr.shape[0], device=ctrl_expr.device)
            v0 = self.velocity_field(ctrl_expr, t0, cond)
            return ctrl_expr + v0
        return self.sample(ctrl_expr, pert_emb)


if __name__ == "__main__":
    # smoke test / usage example
    torch.manual_seed(0)
    B, expr_dim, emb_dim, hidden_dim = 8, 200, 32, 128

    model = VirtualCell(expr_dim, emb_dim, hidden_dim)

    ctrl = torch.randn(B, expr_dim)
    pert = torch.randn(B, expr_dim)
    pert_emb = torch.randn(B, emb_dim)

    # training step
    model.train()
    loss = model.compute_flow_matching_loss(ctrl, pert, pert_emb)
    loss.backward()
    print("flow matching loss:", loss.item())

    # inference (multi-step ODE solve)
    model.eval()
    pred = model(ctrl, pert_emb)
    print("prediction shape:", pred.shape)