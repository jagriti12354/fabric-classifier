"""
Silk vs non-silk fabric classifier (microscope images, transfer learning).

Dataset: "Microscope images of fabric" by Danika Gupta (Kaggle, MIT licence)
  https://www.kaggle.com/datasets/danikagupta/microscope-images-of-fabric
Cite: Gupta, D., Dubey, A. (2024). Towards Improved Sustainability in The Textile
      Lifecycle with Deep Learning. COMPASS '24. https://doi.org/10.1145/3674829.3675077

Runs on: Kaggle Notebook (dataset attached), Google Colab, or local PC with PyTorch.

Usage examples:
  python silk_classifier.py                         # silk vs (cotton + polyester), 5-fold CV
  python silk_classifier.py --mode 3class           # silk / cotton / polyester
  python silk_classifier.py --extra-negatives       # also add nylon, lycra, crepe, jersey, velvet as non-silk
  python silk_classifier.py --predict my_photo.jpg  # predict using saved final model
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Folder names exactly as they appear in the Kaggle dataset
# --------------------------------------------------------------------------
SILK = ["Contrado-Silk"]
COTTON = ["Cotton"]
POLYESTER = ["Polyester"]
# Other pure non-silk fabrics (used only with --extra-negatives).
# Satin and Cotton-Silk-Blend are deliberately left out: satin is a weave
# (fibre content unknown) and the blend contains silk.
EXTRA_NON_SILK = ["Contrado-Nylon", "Contrado - Lycra", "Contrado-Crepe",
                  "Contrado-Jersey", "Contrado-Velvet"]

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Locate dataset
# --------------------------------------------------------------------------
def find_data_root(data_dir=None):
    """Return the directory that directly contains the 14 class folders."""
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    candidates += ["/kaggle/input/microscope-images-of-fabric", "./data"]

    for base in candidates:
        if base and os.path.isdir(base):
            for root, dirs, _ in os.walk(base):
                if "Contrado-Silk" in dirs:
                    return root

    # Fallback: download with kagglehub (Colab / local)
    try:
        import kagglehub
        base = kagglehub.dataset_download("danikagupta/microscope-images-of-fabric")
        for root, dirs, _ in os.walk(base):
            if "Contrado-Silk" in dirs:
                return root
    except Exception as e:  # noqa: BLE001
        print("kagglehub download failed:", e)

    raise FileNotFoundError(
        "Dataset not found. Attach it in the Kaggle notebook, or pass --data-dir "
        "pointing to the unzipped folder.")


def collect_samples(root, mode, extra_negatives):
    """Return list of (path, label) and class names."""
    if mode == "binary":
        groups = {"non-silk": COTTON + POLYESTER + (EXTRA_NON_SILK if extra_negatives else []),
                  "silk": SILK}
        class_names = ["non-silk", "silk"]
    else:  # 3class
        groups = {"cotton": COTTON, "polyester": POLYESTER, "silk": SILK}
        class_names = ["cotton", "polyester", "silk"]

    samples = []
    for label, cname in enumerate(class_names):
        for folder in groups[cname]:
            fdir = os.path.join(root, folder)
            if not os.path.isdir(fdir):
                print(f"  [warn] folder missing: {folder}")
                continue
            files = sorted(f for f in os.listdir(fdir) if f.lower().endswith(IMG_EXT))
            samples += [(os.path.join(fdir, f), label) for f in files]
            print(f"  {folder:<20} -> {cname:<10} {len(files)} images")
    return samples, class_names


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_CACHE = {}


def load_image(path, short_side=320):
    """Decode once and keep a downsized copy in memory (originals are ~1.6 MB JPEGs)."""
    if path not in _CACHE:
        img = Image.open(path).convert("RGB")
        img.thumbnail((short_side * 4, short_side), Image.BILINEAR) if img.height < img.width \
            else img.thumbnail((short_side, short_side * 4), Image.BILINEAR)
        _CACHE[path] = img
    return _CACHE[path]


class ImageList(Dataset):
    def __init__(self, samples, tfm):
        self.samples, self.tfm = samples, tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tfm(load_image(path)), label


def get_transforms(img_size):
    # Fabric texture has no fixed orientation -> flips/rotations are safe.
    # Colour jitter is kept mild so the model can't rely only on dye colour.
    train = transforms.Compose([
        transforms.Resize(int(img_size * 1.25)),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test = transforms.Compose([
        transforms.Resize(int(img_size * 1.25)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, test


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(arch, n_classes, pretrained=True):
    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds += out.argmax(1).cpu().tolist()
            trues += y.cpu().tolist()
    return total_loss / len(trues), np.array(trues), np.array(preds)


def train_model(train_samples, val_samples, n_classes, args, device):
    tf_train, tf_test = get_transforms(args.img_size)
    tl = DataLoader(ImageList(train_samples, tf_train), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = None
    if val_samples:
        vl = DataLoader(ImageList(val_samples, tf_test), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, n_classes).to(device)
    w = class_weights([l for _, l in train_samples], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, best_pred = -1, None, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(model, tl, device, criterion, optimizer)
        scheduler.step()
        msg = f"    ep {ep:02d} train loss {tr_loss:.3f} acc {accuracy_score(tr_y, tr_p):.3f}"
        if vl is not None:
            va_loss, va_y, va_p = run_epoch(model, vl, device, criterion)
            f1 = f1_score(va_y, va_p, average="macro")
            msg += f" | val loss {va_loss:.3f} acc {accuracy_score(va_y, va_p):.3f} macroF1 {f1:.3f}"
            # NOTE: picking the best epoch on the val fold slightly inflates CV scores.
            # For a strict estimate use a fixed epoch count (--no-best-epoch).
            if args.no_best_epoch or f1 >= best_f1:
                best_f1, best_pred = f1, (va_y, va_p)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg + f" ({time.time() - t0:.0f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_pred


# --------------------------------------------------------------------------
# Cross-validation + final model
# --------------------------------------------------------------------------
def cross_validate(samples, class_names, args, device):
    X = np.arange(len(samples))
    y = np.array([l for _, l in samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_true, all_pred, fold_acc, fold_f1 = [], [], [], []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {k}/{args.folds}  (train {len(tr)}, val {len(va)}) ===")
        _, (vy, vp) = train_model([samples[i] for i in tr], [samples[i] for i in va],
                                  len(class_names), args, device)
        fold_acc.append(accuracy_score(vy, vp))
        fold_f1.append(f1_score(vy, vp, average="macro"))
        all_true += vy.tolist()
        all_pred += vp.tolist()

    print("\n================ CROSS-VALIDATION SUMMARY ================")
    print(f"Accuracy : {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
    print(f"Macro F1 : {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
    print("\nPooled classification report:")
    print(classification_report(all_true, all_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    print("            " + "  ".join(f"{c:>10}" for c in class_names))
    for cname, row in zip(class_names, cm):
        print(f"{cname:>10}  " + "  ".join(f"{v:>10}" for v in row))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cv_results.json"), "w") as f:
        json.dump({"fold_acc": fold_acc, "fold_f1": fold_f1, "classes": class_names,
                   "confusion_matrix": cm.tolist(), "args": vars(args)}, f, indent=2)
    save_confusion_plot(cm, class_names, os.path.join(args.out, "confusion_matrix.png"))


def save_confusion_plot(cm, class_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4 + len(class_names), 3.5 + len(class_names) * 0.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Pooled CV confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def train_final(samples, class_names, args, device):
    print("\n=== Training final model on ALL images ===")
    model, _ = train_model(samples, None, len(class_names), args, device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "silk_model.pt")
    torch.save({"state_dict": model.state_dict(), "classes": class_names,
                "arch": args.arch, "img_size": args.img_size}, path)
    print(f"Saved {path}")


def predict(image_paths, args, device):
    ckpt = torch.load(os.path.join(args.out, "silk_model.pt"), map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _, tf = get_transforms(ckpt["img_size"])
    for p in image_paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        best = int(prob.argmax())
        detail = ", ".join(f"{c}={v:.2f}" for c, v in zip(ckpt["classes"], prob))
        print(f"{p}: {ckpt['classes'][best]}  ({detail})")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Silk vs non-silk microscope classifier")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--mode", choices=["binary", "3class"], default="binary")
    ap.add_argument("--extra-negatives", action="store_true",
                    help="binary mode: add nylon/lycra/crepe/jersey/velvet to non-silk")
    ap.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-best-epoch", action="store_true",
                    help="report last-epoch scores instead of best-epoch (stricter CV)")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--predict", nargs="+", default=None, help="image(s) to classify")
    args, _ = ap.parse_known_args()  # parse_known_args: safe inside notebooks

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.predict:
        predict(args.predict, args, device)
        return

    root = find_data_root(args.data_dir)
    print("Data root:", root)
    samples, class_names = collect_samples(root, args.mode, args.extra_negatives)
    print(f"Total {len(samples)} images, classes = {class_names}")

    if not args.skip_cv:
        cross_validate(samples, class_names, args, device)
    if not args.skip_final:
        train_final(samples, class_names, args, device)


if __name__ == "__main__":
    main()
