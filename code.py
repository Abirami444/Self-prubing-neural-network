import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ──────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
# PrunableLinear  (v3 — gate init and threshold fixed)
# ══════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Linear layer with per-weight learnable gates.

    Key design decisions (v3):
    ─────────────────────────
    gate_scores init = +3.0
        sigmoid(+3) ≈ 0.95  →  all gates start nearly open (active).
        L1 pressure then closes unneeded gates over training.
        sigmoid(0) = 0.5 was the bug in v2: gradient is maximally
        ambiguous at 0.5 — the gate has no "default" direction to go.
        Starting high and letting L1 pull gates down is the standard
        approach used in learned pruning literature (e.g. L0 regularization,
        Louizos et al. 2018).

    Training  → soft gate  = sigmoid(score)          [differentiable]
    Inference → hard gate  = sigmoid(score) > thresh  [true pruning]
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.threshold    = threshold   # gates below this → truly pruned

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # FIX A: init to +3.0 so gates start at ~0.95 (open), not 0.5
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 3.0)
        )

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)

        if self.training:
            pruned_weights = self.weight * gates          # soft — gradients flow
        else:
            binary_gates   = (gates > self.threshold).float()
            pruned_weights = self.weight * binary_gates   # hard — true pruning

        return F.linear(x, pruned_weights, self.bias)

    def get_soft_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def get_binary_gates(self) -> torch.Tensor:
        return (torch.sigmoid(self.gate_scores) > self.threshold).float().detach()

    def sparsity_level(self) -> float:
        """Fraction of weights truly pruned (hard gate = 0)."""
        return (self.get_binary_gates() == 0).float().mean().item()

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"threshold={self.threshold}")


# ══════════════════════════════════════════════════════════════════
# Network: Conv feature extractor + Prunable FC head
# ══════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    CNN feature extractor (standard Conv layers) +
    Prunable FC classification head.

    Conv layers learn spatial features.
    PrunableLinear layers learn WHICH features actually matter for
    classification — redundant feature combinations get pruned out.

    Architecture:
        (3,32,32) → Conv(32) → Conv(64) → Conv(128) → flatten(2048)
                 → PrunableLinear(2048→512) → BN → ReLU → Dropout
                 → PrunableLinear(512→10)
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        # output: 128 × 4 × 4 = 2048

        self.fc1  = PrunableLinear(2048, 512, threshold=threshold)
        self.bn1  = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=0.4)
        self.fc2  = PrunableLinear(512, 10, threshold=threshold)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        return self.fc2(x)

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        FIX C: mean of all soft gate values.

        Range: (0, 1) always, regardless of model size.
        Cross-entropy for a 10-class problem ≈ 1.5–2.5.
        With λ=0.3, sparsity contributes ~0.3×0.9 ≈ 0.27 initially —
        meaningful pressure from epoch 1 without overwhelming classification.
        """
        all_gates = [
            torch.sigmoid(layer.gate_scores)
            for layer in self.prunable_layers()
        ]
        return torch.cat([g.reshape(-1) for g in all_gates]).mean()

    def overall_sparsity(self) -> float:
        total, pruned = 0, 0
        for layer in self.prunable_layers():
            bg = layer.get_binary_gates()
            pruned += (bg == 0).sum().item()
            total  += bg.numel()
        return pruned / total if total > 0 else 0.0

    def all_soft_gate_values(self) -> np.ndarray:
        return np.concatenate([
            layer.get_soft_gates().cpu().numpy().ravel()
            for layer in self.prunable_layers()
        ])

    def weight_parameters(self):
        """Yields (name, param) for non-gate parameters."""
        for name, p in self.named_parameters():
            if "gate_scores" not in name:
                yield p

    def gate_parameters(self):
        """Yields gate_scores parameters only."""
        for name, p in self.named_parameters():
            if "gate_scores" in name:
                yield p


# ══════════════════════════════════════════════════════════════════
# Compressed model builder (Fix 4)
# ══════════════════════════════════════════════════════════════════

def create_compressed_model(model: SelfPruningNet) -> nn.Module:
    """
    Physically remove pruned weights after training.
    Returns a plain nn.Sequential with only surviving connections.
    """
    model.eval()
    layers_list = list(model.prunable_layers())
    compressed  = []
    prev_kept   = None

    for i, layer in enumerate(layers_list):
        with torch.no_grad():
            bg     = layer.get_binary_gates()          # (out, in)
            w      = layer.weight.data.clone() * bg    # zeroed pruned weights
            b      = layer.bias.data.clone()

            if prev_kept is not None:
                w = w[:, prev_kept]

            is_last   = (i == len(layers_list) - 1)
            if is_last:
                active = torch.ones(w.size(0), dtype=torch.bool)
            else:
                active = (bg.sum(dim=1) > 0)

            kept_idx = active.nonzero(as_tuple=True)[0]
            w = w[kept_idx]
            b = b[kept_idx]

            new_fc = nn.Linear(w.size(1), w.size(0))
            new_fc.weight.data = w
            new_fc.bias.data   = b
            compressed.append(new_fc)

            pct = kept_idx.numel() / bg.size(0) * 100
            print(f"  FC{i+1}: {bg.size()} → "
                  f"({kept_idx.numel()} × {w.size(1)})  "
                  f"[{pct:.1f}% neurons retained]")
            prev_kept = kept_idx

    final = []
    for idx, fc in enumerate(compressed):
        final.append(fc)
        if idx < len(compressed) - 1:
            final.append(nn.ReLU(inplace=True))

    return nn.Sequential(*final)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def get_loaders(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    use_gpu   = torch.cuda.is_available()
    n_workers = 2 if use_gpu else 0

    train_ds = torchvision.datasets.CIFAR10("./data", True,  download=True, transform=tr)
    test_ds  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=te)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=n_workers, pin_memory=use_gpu)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=n_workers, pin_memory=use_gpu)
    return train_ld, test_ld


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = correct = n = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)

        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()            # always in (0,1)
        loss        = cls_loss + lambda_sparse * sparse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return correct / n


def train_model(lambda_sparse, epochs, device, train_ld, test_ld):
    print(f"\n{'='*58}")
    print(f"  Training  λ = {lambda_sparse}  ({epochs} epochs)")
    print(f"{'='*58}")

    model = SelfPruningNet(threshold=0.5).to(device)

    # FIX B: Separate learning rates for weights vs gates
    #   Weights: lr=1e-3 (standard)
    #   Gates:   lr=5e-3 (faster — need to converge to 0 or 1 decisively)
    optimizer = optim.Adam([
        {"params": list(model.weight_parameters()), "lr": 1e-3},
        {"params": list(model.gate_parameters()),   "lr": 5e-3},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_ld, optimizer, lambda_sparse, device
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            sparsity = model.overall_sparsity()
            test_acc = evaluate(model, test_ld, device)
            model.train()

            # Show mean gate value so we can see gates moving toward 0
            mean_gate = model.all_soft_gate_values().mean()
            print(
                f"  Ep {epoch:>3}/{epochs}  "
                f"loss={train_loss:.3f}  "
                f"train={train_acc*100:.1f}%  "
                f"test={test_acc*100:.1f}%  "
                f"sparsity={sparsity*100:.1f}%  "
                f"mean_gate={mean_gate:.3f}"
            )

    final_acc      = evaluate(model, test_ld, device)
    final_sparsity = model.overall_sparsity()
    mean_gate      = model.all_soft_gate_values().mean()

    print(f"\n  ✓ Test Accuracy  : {final_acc*100:.2f}%")
    print(f"  ✓ Sparsity Level : {final_sparsity*100:.2f}%  (binary, threshold=0.5)")
    print(f"  ✓ Mean Gate Value: {mean_gate:.4f}  (lower = more pruning pressure)")
    return final_acc, final_sparsity, model


# ══════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════

def plot_gate_distribution(model, lambda_val, save_path):
    gates = model.all_soft_gate_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Gate Value Distribution  |  λ = {lambda_val}",
                 fontsize=13, fontweight="bold")

    axes[0].hist(gates, bins=100, color="#4C72B0", edgecolor="white", lw=0.3)
    axes[0].axvline(0.5, color="red", ls="--", label="Hard threshold (0.5)")
    axes[0].set_xlabel("Soft Gate Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Full distribution")
    axes[0].legend()

    low = gates[gates < 0.6]
    axes[1].hist(low, bins=80, color="#DD8452", edgecolor="white", lw=0.3)
    axes[1].axvline(0.5, color="red", ls="--", label="Hard threshold (0.5)")
    axes[1].set_xlabel("Soft Gate Value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Zoomed: gates < 0.6  (pruned spike on left)")
    axes[1].legend()

    pct = (gates < 0.5).mean() * 100
    fig.text(0.5, 0.01,
             f"Binary-pruned gates: {pct:.1f}%  |  Total: {len(gates):,}",
             ha="center", fontsize=10, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tradeoff(results, save_path):
    lambdas    = [r["lambda"] for r in results]
    accs       = [r["accuracy"] * 100 for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("λ (Sparsity Penalty)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", color="#2196F3", fontsize=11)
    ax1.plot(lambdas, accs, "o-", color="#2196F3", lw=2, ms=8, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity Level (%)", color="#F44336", fontsize=11)
    ax2.plot(lambdas, sparsities, "s--", color="#F44336", lw=2, ms=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left")
    plt.title("Accuracy vs Sparsity Trade-off", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    # FIX D: λ values meaningful relative to cross-entropy scale (~1.5–2.5)
    # sparsity_loss is now always in (0,1), so:
    #   λ=0.1 → mild pruning pressure
    #   λ=0.3 → moderate (best trade-off expected)
    #   λ=0.6 → aggressive pruning
    EPOCHS      = 40
    BATCH_SIZE  = 128
    LAMBDAS     = [0.1, 0.3, 0.6]
    BEST_LAMBDA = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Seed   : {SEED}")
    print(f"Lambdas: {LAMBDAS}  (rescaled — sparsity loss is now in (0,1))")

    os.makedirs("outputs", exist_ok=True)
    train_ld, test_ld = get_loaders(BATCH_SIZE)

    results    = []
    best_model = None

    for lam in LAMBDAS:
        acc, sparsity, model = train_model(lam, EPOCHS, device, train_ld, test_ld)
        results.append({"lambda": lam, "accuracy": acc, "sparsity": sparsity})
        if lam == BEST_LAMBDA:
            best_model = model

    # ── Summary ────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<10} {'Test Accuracy':>15} {'Sparsity (binary)':>20}")
    print(f"  {'-'*10} {'-'*15} {'-'*20}")
    for r in results:
        print(f"  {r['lambda']:<10}  {r['accuracy']*100:>13.2f}%  {r['sparsity']*100:>18.2f}%")
    print("="*60)

    # ── Compressed model ───────────────────────────────────────
    if best_model is not None:
        print(f"\n── Compressed model  (λ={BEST_LAMBDA}) ──")
        create_compressed_model(best_model)
        plot_gate_distribution(
            best_model, BEST_LAMBDA,
            save_path="outputs/gate_distribution.png"
        )

    plot_tradeoff(results, save_path="outputs/accuracy_sparsity_tradeoff.png")
    print("\nDone. Check outputs/ folder.")


if __name__ == "__main__":
    main()




