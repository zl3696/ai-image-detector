import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 0.001
NUM_WORKERS = 2
 
# Point this at your local CIFAKE root, which should contain:
#   train/REAL/, train/FAKE/, test/REAL/, test/FAKE/
DATA_ROOT = "./cifake"
 
print(f"Running on: {DEVICE}")
 
 
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
 
def get_loaders(resize: int) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for a given input resolution."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
 
    train_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
 
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=test_tf)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
 
    return train_loader, test_loader
 
 
def build_resnet(unfreeze: str = "head_only") -> nn.Module:
    """
    Build a ResNet-18 with ImageNet weights.
 
    unfreeze options:
        "head_only"   → freeze everything, replace & train only the FC head
        "layer4"      → also unfreeze layer4
        "layer3_4"    → also unfreeze layer3 + layer4
        "all"         → full fine-tune, nothing frozen
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
 
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
 
    # Selectively unfreeze
    unfreeze_map = {
        "head_only": [],
        "layer4":    [model.layer4],
        "layer3_4":  [model.layer3, model.layer4],
        "all":       [model.layer1, model.layer2, model.layer3, model.layer4,
                      model.bn1, model.conv1],
    }
    for block in unfreeze_map[unfreeze]:
        for param in block.parameters():
            param.requires_grad = True
 
    # Always replace & train the classification head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
 
    return model.to(DEVICE)
 
 
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
 
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
 
    return running_loss / len(loader.dataset)
 
 
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
 
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
 
        preds     = model(imgs)
        val_loss += criterion(preds, labels).item() * imgs.size(0)
 
        binary = (preds >= 0.5).long().squeeze(1).cpu().numpy()
        all_preds.extend(binary)
        all_labels.extend(labels.long().squeeze(1).cpu().numpy())
 
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1       = f1_score(all_labels, all_preds, average="macro")
    avg_loss = val_loss / len(loader.dataset)
 
    return accuracy, f1, avg_loss, all_preds, all_labels
 
 
def run_experiment(label: str, model: nn.Module, train_loader: DataLoader,
                   test_loader: DataLoader) -> dict:
    """Train a model and return a results dict including per-epoch curves."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
 
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
 
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
    print(f"{'='*60}")
 
    best_acc   = 0.0
    best_state = None
 
    for epoch in range(1, EPOCHS + 1):
        train_loss            = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1, val_loss, _, _ = evaluate(model, test_loader, criterion)
 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
 
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}")
 
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())
 
    # Final evaluation with best weights
    model.load_state_dict(best_state)
    final_acc, final_f1, _, preds, labels = evaluate(model, test_loader, criterion)
    report = classification_report(labels, preds, target_names=["REAL", "FAKE"])
    cm     = confusion_matrix(labels, preds)
 
    print(f"\n  Final accuracy : {final_acc:.4f}")
    print(f"  Final F1 (macro): {final_f1:.4f}")
    print(f"\n{report}")
 
    return {
        "label":    label,
        "accuracy": final_acc,
        "f1":       final_f1,
        "report":   report,
        "cm":       cm.tolist(),
        "history":  history,
    }
 
 
# ─────────────────────────────────────────────
# Experiment A — Resolution
# ─────────────────────────────────────────────
 
def experiment_a():
    """
    Hold the backbone frozen (head only). Compare 32×32 vs 224×224 input.
    This isolates resolution as the variable.
    """
    results = []
 
    for res in [32, 224]:
        train_loader, test_loader = get_loaders(res)
        model = build_resnet(unfreeze="head_only")
        r = run_experiment(f"ResNet-18 (frozen) @ {res}×{res}", model, train_loader, test_loader)
        r["resolution"] = res
        results.append(r)
 
    return results
 
 
# ─────────────────────────────────────────────
# Experiment B — Fine-tuning depth
# ─────────────────────────────────────────────
 
def experiment_b():
    """
    Fix resolution at 224×224. Vary how many backbone layers are unfrozen.
    This isolates fine-tuning depth as the variable.
    """
    configs = [
        ("head_only",  "Frozen backbone (head only)"),
        ("layer4",     "Unfreeze layer4"),
        ("layer3_4",   "Unfreeze layer3 + layer4"),
        ("all",        "Full fine-tune"),
    ]
 
    train_loader, test_loader = get_loaders(224)
    results = []
 
    for unfreeze_key, label in configs:
        model = build_resnet(unfreeze=unfreeze_key)
        r = run_experiment(label, model, train_loader, test_loader)
        r["unfreeze"] = unfreeze_key
        results.append(r)
 
    return results
 
 
# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
 
def plot_experiment_a(results_a: list[dict]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Experiment A — Resolution Ablation (ResNet-18, frozen backbone)", fontsize=13)
 
    colors = ["#4C72B0", "#DD8452"]
 
    # Training curves
    ax = axes[0]
    for r, c in zip(results_a, colors):
        ax.plot(r["history"]["val_acc"], label=f'{r["resolution"]}×{r["resolution"]}', color=c)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Val loss
    ax = axes[1]
    for r, c in zip(results_a, colors):
        ax.plot(r["history"]["val_loss"], label=f'{r["resolution"]}×{r["resolution"]}', color=c)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Final accuracy bar
    ax = axes[2]
    labels = [f'{r["resolution"]}×{r["resolution"]}' for r in results_a]
    accs   = [r["accuracy"] for r in results_a]
    bars   = ax.bar(labels, accs, color=colors, width=0.4)
    ax.set_title("Final Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("experiment_a_resolution.png", dpi=150, bbox_inches="tight")
    print("\nSaved: experiment_a_resolution.png")
    plt.show()
 
 
def plot_experiment_b(results_b: list[dict]):
    labels = [r["label"] for r in results_b]
    accs   = [r["accuracy"] for r in results_b]
    f1s    = [r["f1"] for r in results_b]
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment B — Fine-tuning Depth Ablation (ResNet-18 @ 224×224)", fontsize=13)
 
    palette = sns.color_palette("Blues_d", len(results_b))
 
    # Accuracy by depth
    ax = axes[0]
    bars = ax.bar(range(len(labels)), accs, color=palette)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Final Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
 
    # F1 by depth
    ax = axes[1]
    bars = ax.bar(range(len(labels)), f1s, color=palette)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Final Macro F1")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.5, 1.0)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
 
    # Val accuracy training curves
    ax = axes[2]
    for r, c in zip(results_b, palette):
        ax.plot(r["history"]["val_acc"], label=r["label"], color=c)
    ax.set_title("Validation Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("experiment_b_finetuning.png", dpi=150, bbox_inches="tight")
    print("Saved: experiment_b_finetuning.png")
    plt.show()
 
 
# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
 
def print_summary(results_a, results_b):
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
 
    print("\nExperiment A — Resolution (frozen backbone):")
    print(f"  {'Config':<30} {'Accuracy':>10} {'F1 (macro)':>12}")
    print(f"  {'-'*52}")
    for r in results_a:
        print(f"  {r['label']:<30} {r['accuracy']:>10.4f} {r['f1']:>12.4f}")
 
    print("\nExperiment B — Fine-tuning depth (224×224):")
    print(f"  {'Config':<35} {'Accuracy':>10} {'F1 (macro)':>12}")
    print(f"  {'-'*57}")
    for r in results_b:
        print(f"  {r['label']:<35} {r['accuracy']:>10.4f} {r['f1']:>12.4f}")
 
    print("\nInterpretation guide:")
    print("  • If Exp A shows big jump (32→224): resolution was the bottleneck.")
    print("  • If Exp B shows increasing accuracy with depth: frozen backbone was the bottleneck.")
    print("  • Both effects can be real and additive.")
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("Running Experiment A (Resolution)...")
    results_a = experiment_a()
 
    print("\nRunning Experiment B (Fine-tuning depth)...")
    results_b = experiment_b()
 
    print_summary(results_a, results_b)
    plot_experiment_a(results_a)
    plot_experiment_b(results_b)
 
    # Save raw results for blog post
    serializable = {
        "experiment_a": [
            {k: v for k, v in r.items() if k != "cm"} for r in results_a
        ],
        "experiment_b": [
            {k: v for k, v in r.items() if k != "cm"} for r in results_b
        ],
    }
    with open("ablation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("\nRaw results saved to ablation_results.json")
