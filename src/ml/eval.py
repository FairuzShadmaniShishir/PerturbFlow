import torch
import torch.nn.functional as F


# Loss
def loss_fn(pred_expr, target_expr, ctrl_expr, l1_weight=0.3, cos_weight=1.0):
    l1_loss = F.l1_loss(pred_expr, target_expr)
    pred_diff = pred_expr - ctrl_expr
    true_diff = target_expr - ctrl_expr
    cos_loss = 1 - F.cosine_similarity(pred_diff, true_diff, dim=1).mean()

    return l1_weight * l1_loss + cos_weight * cos_loss


def pearson_diff_corr(pred, target, ctrl):
    pred_diff = pred - ctrl
    target_diff = target - ctrl

    pred_centered = pred_diff - pred_diff.mean(dim=1, keepdim=True)
    target_centered = target_diff - target_diff.mean(dim=1, keepdim=True)

    return F.cosine_similarity(pred_centered, target_centered).mean().item()


def compute_baseline_metrics(val_loader, mean_expr, device):
    """Compute baseline loss and correlation using mean train perturbation as prediction."""
    baseline_loss = 0.0
    baseline_corr = 0.0
    with torch.no_grad():
        for batch_ctrl_expr, _, batch_target_expr in val_loader:
            batch_ctrl_expr = batch_ctrl_expr.to(device)
            batch_target_expr = batch_target_expr.to(device)
            baseline_loss += loss_fn(mean_expr.expand_as(batch_target_expr), batch_target_expr, batch_ctrl_expr).item() * batch_target_expr.size(0)
            baseline_corr += pearson_diff_corr(mean_expr.expand_as(batch_target_expr), batch_target_expr, batch_ctrl_expr) * batch_target_expr.size(0)
    baseline_loss /= len(val_loader.dataset)
    baseline_corr /= len(val_loader.dataset)
    return baseline_loss, baseline_corr