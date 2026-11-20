"""
Cotton vs Polyester identification + blend % prediction from NIR spectra
Dataset: NIST NIR-SORT (Benchtop_N, Nicolet iS50 FT-NIR, 833-2500 nm)
  https://data.nist.gov/od/id/mds2-3325   DOI 10.18126/61v0-rf98
Cite: Goodge K.E. et al., "A spectroscopic dataset for known provenance and
      post-consumer textiles", Scientific Data (2026).

What it does
  1. Downloads the benchtop absorbance spectra (~26 MB) if not already present
  2. Labels each spectrum from its file name (fibre code + % is in the name)
  3. Preprocessing: wavenumber -> nm, common grid, SNV, Savitzky-Golay derivative
  4. Task A: pure cotton vs pure polyester classification (PLS-DA and SVM)
  5. Task B: cotton % in cotton/polyester blends (PLS regression)
  All CV is GROUPED BY FABRIC, so replicates of one fabric never sit in both
  train and test (otherwise accuracy is falsely ~100%).

Run:
  pip install numpy scipy scikit-learn matplotlib
  python nir_cotton_polyester.py
  python nir_cotton_polyester.py --data reflectance --deriv 2
"""

import argparse
import io
import os
import re
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = "https://data.nist.gov/od/ds/ark:/88434/mds2-3325"
SOURCES = ["A", "B", "C", "D"]

# Fibre letter codes used in NIR-SORT file names (from SampleInformation.csv)
FIBRE_CODE = {"a": "cotton", "b": "polyester", "c": "nylon", "d": "acrylic",
              "e": "spandex", "f": "recycled polyester", "g": "rayon",
              "x": None}
POLYESTER_CODES = {"b", "f"}   # recycled PET is chemically the same polymer

# Bands to mark on plots (approximate, nm)
BANDS = {1490: "O-H (cellulose)", 1660: "Ar C-H (PET)", 1930: "H2O / O-H",
         2100: "O-H/C-O (cellulose)", 2140: "C=O comb. (PET)"}

NAME_RE = re.compile(r"([A-D]\d+)_((?:[a-z]\d{2,3}){3})_(\d\d)_(\d\d)_N_(\d+)([AR])\.CSV$",
                     re.IGNORECASE)


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def download(data_dir, kind):
    sub = "Absorbance" if kind == "absorbance" else "Reflectance"
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for s in SOURCES:
        dst = os.path.join(data_dir, f"Benchtop_N_{sub}_Source_{s}.zip")
        if not os.path.exists(dst):
            url = f"{BASE}/Spectroscopy/NearInfrared/Benchtop_N/{sub}/Source_{s}/Source_{s}.zip"
            print("Downloading", url)
            urllib.request.urlretrieve(url, dst)
        paths.append(dst)
    return paths


# --------------------------------------------------------------------------
# 2. Load + label
# --------------------------------------------------------------------------
def parse_composition(code):
    """'a065b035x000' -> {'cotton': 65, 'polyester': 35}"""
    comp = {}
    for letter, pct in re.findall(r"([a-z])(\d{2,3})", code):
        if letter == "x" or int(pct) == 0:
            continue
        name = "polyester" if letter in POLYESTER_CODES else FIBRE_CODE.get(letter, f"code_{letter}")
        comp[name] = comp.get(name, 0) + int(pct)
    return comp


def load_spectra(zip_paths, grid):
    X, meta = [], []
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                m = NAME_RE.search(name.split("/")[-1])
                if not m:   # skips stand-off-distance files (instrument Q) etc.
                    continue
                sample_id, code, form = m.group(1), m.group(2), m.group(4)
                arr = np.loadtxt(io.BytesIO(z.read(name)), delimiter=",")
                wn, y = arr[:, 0], arr[:, 1]
                nm = 1e7 / wn
                order = np.argsort(nm)
                X.append(np.interp(grid, nm[order], y[order]))
                meta.append({"sample": sample_id, "code": code,
                             "form": {"00": "woven", "01": "knit", "04": "fibre"}.get(form, form),
                             "comp": parse_composition(code)})
    return np.array(X), meta


# --------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------
def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def preprocess(X, deriv, window):
    Xp = snv(X)
    if deriv > 0:
        Xp = savgol_filter(Xp, window_length=window, polyorder=2, deriv=deriv, axis=1)
    return Xp


# --------------------------------------------------------------------------
# 4. Task A – classification
# --------------------------------------------------------------------------
class PLSDA:
    """Two-class PLS-DA: regress 0/1 and threshold at 0.5."""
    def __init__(self, n_components=5):
        self.pls = PLSRegression(n_components=n_components, scale=False)

    def fit(self, X, y):
        self.pls.fit(X, y.astype(float))
        return self

    def predict(self, X):
        return (self.pls.predict(X).ravel() > 0.5).astype(int)


def grouped_cv_predict(make_model, X, y, groups, n_splits):
    n_groups = len(set(groups))
    cv = LeaveOneGroupOut() if n_groups <= n_splits else GroupKFold(n_splits=n_splits)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y, groups):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def task_classification(Xp, meta, args, grid, Xraw, out):
    idx = [i for i, m in enumerate(meta)
           if m["comp"] in ({"cotton": 100}, {"polyester": 100})]
    if args.exclude_fibre_form:
        idx = [i for i in idx if meta[i]["form"] != "fibre"]
    X = Xp[idx]
    y = np.array([1 if meta[i]["comp"].get("polyester") else 0 for i in idx])
    groups = np.array([meta[i]["sample"] for i in idx])
    names = ["cotton", "polyester"]

    n_fab = {c: len({g for g, l in zip(groups, y) if l == k}) for k, c in enumerate(names)}
    print("\n================ TASK A: cotton vs polyester ================")
    print(f"Spectra: {len(y)}  | fabrics: cotton {n_fab['cotton']}, polyester {n_fab['polyester']}")

    models = {
        f"PLS-DA ({args.lv} LV)": lambda: PLSDA(args.lv),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(),
                                           SVC(C=10, gamma="scale", class_weight="balanced")),
    }
    for mname, mk in models.items():
        pred = grouped_cv_predict(mk, X, y, groups, args.folds)
        print(f"\n--- {mname}  (grouped CV, {args.folds} folds) ---")
        print(f"Spectrum-level accuracy: {accuracy_score(y, pred):.3f}")
        # fabric-level: majority vote over replicates
        fab_true, fab_pred = [], []
        for g in np.unique(groups):
            mask = groups == g
            fab_true.append(y[mask][0])
            fab_pred.append(int(pred[mask].mean() > 0.5))
        print(f"Fabric-level accuracy (majority vote): {accuracy_score(fab_true, fab_pred):.3f}"
              f"  ({sum(np.array(fab_true) == np.array(fab_pred))}/{len(fab_true)} fabrics)")
        print(classification_report(y, pred, target_names=names, digits=3))
        print("Confusion matrix (rows true, cols pred):\n", confusion_matrix(y, pred))
        wrong = sorted({groups[i] for i in np.where(pred != y)[0]})
        if wrong:
            print("Fabrics with any misclassified replicate:", ", ".join(wrong))

    plot_mean_spectra(grid, Xraw[idx], Xp[idx], y, names, args, out)
    plot_pca(X, y, names, out)


# --------------------------------------------------------------------------
# 5. Task B – blend regression
# --------------------------------------------------------------------------
def task_blend(Xp, meta, args, out):
    idx = [i for i, m in enumerate(meta)
           if set(m["comp"]) <= {"cotton", "polyester"} and sum(m["comp"].values()) == 100]
    if args.exclude_fibre_form:
        idx = [i for i in idx if meta[i]["form"] != "fibre"]
    X = Xp[idx]
    y = np.array([meta[i]["comp"].get("cotton", 0) for i in idx], dtype=float)
    groups = np.array([meta[i]["sample"] for i in idx])
    n_groups = len(set(groups))

    print("\n================ TASK B: cotton % in cotton/PET blends ================")
    print(f"Spectra: {len(y)} | specimens: {n_groups} | cotton % levels: "
          f"{sorted(int(v) for v in set(y))}")

    cv = GroupKFold(n_splits=min(args.folds, n_groups))
    rmse_by_lv = {}
    best = None
    for lv in range(1, args.max_lv + 1):
        pred = np.empty_like(y)
        for tr, te in cv.split(X, y, groups):
            pls = PLSRegression(n_components=lv, scale=False).fit(X[tr], y[tr])
            pred[te] = pls.predict(X[te]).ravel()
        pred = np.clip(pred, 0, 100)
        rmse = np.sqrt(mean_squared_error(y, pred))
        rmse_by_lv[lv] = rmse
        if best is None or rmse < best[1]:
            best = (lv, rmse, pred.copy())
    print("RMSECV by number of latent variables:")
    print("  " + "  ".join(f"{k}:{v:.1f}" for k, v in rmse_by_lv.items()))
    lv, rmse, pred = best
    print(f"Best: {lv} LV  ->  RMSECV = {rmse:.2f} % cotton,  R2 = {r2_score(y, pred):.3f}")
    print("(Choosing LV on the same CV is slightly optimistic; report it as such.)")
    plot_blend(y, pred, rmse_by_lv, lv, out)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mean_spectra(grid, Xraw, Xp, y, names, args, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for k, c in enumerate(names):
        axes[0].plot(grid, Xraw[y == k].mean(0), label=c)
        axes[1].plot(grid, Xp[y == k].mean(0), label=c)
    for ax in axes:
        for b, lab in BANDS.items():
            ax.axvline(b, color="grey", ls=":", lw=0.8)
            ax.text(b, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7)
        ax.legend()
    axes[0].set_ylabel(f"{args.data} (raw)")
    axes[1].set_ylabel(f"SNV + SG d{args.deriv}")
    axes[1].set_xlabel("Wavelength (nm)")
    fig.suptitle("Mean NIR spectra – NIR-SORT benchtop")
    fig.tight_layout()
    p = os.path.join(out, "mean_spectra.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


def plot_pca(X, y, names, out):
    plt = _plt()
    pca = PCA(n_components=2).fit(X)
    S = pca.transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, c in enumerate(names):
        ax.scatter(S[y == k, 0], S[y == k, 1], s=12, label=c, alpha=0.7)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.legend()
    ax.set_title("PCA scores – pure cotton vs polyester")
    fig.tight_layout()
    p = os.path.join(out, "pca_scores.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


def plot_blend(y, pred, rmse_by_lv, lv, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(list(rmse_by_lv), list(rmse_by_lv.values()), "o-")
    axes[0].axvline(lv, color="red", ls="--")
    axes[0].set_xlabel("PLS latent variables")
    axes[0].set_ylabel("RMSECV (% cotton)")
    axes[1].scatter(y, pred, s=12, alpha=0.6)
    axes[1].plot([0, 100], [0, 100], "k--", lw=1)
    axes[1].set_xlabel("Label cotton %")
    axes[1].set_ylabel("Predicted cotton %")
    axes[1].set_title(f"PLS ({lv} LV), grouped CV")
    fig.tight_layout()
    p = os.path.join(out, "blend_regression.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NIR-SORT cotton vs polyester")
    ap.add_argument("--data-dir", default="nirsort_data")
    ap.add_argument("--data", choices=["absorbance", "reflectance"], default="absorbance")
    ap.add_argument("--wl-min", type=float, default=1100)
    ap.add_argument("--wl-max", type=float, default=2450)
    ap.add_argument("--step", type=float, default=2.0, help="nm step of common grid")
    ap.add_argument("--deriv", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--window", type=int, default=15, help="SG window (points on grid)")
    ap.add_argument("--lv", type=int, default=5, help="PLS-DA latent variables")
    ap.add_argument("--max-lv", type=int, default=12, help="max LV for blend PLS")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--exclude-fibre-form", action="store_true",
                    help="drop loose-fibre specimens (form 04); keep fabrics only")
    ap.add_argument("--out", default="nir_outputs")
    args, _ = ap.parse_known_args()

    os.makedirs(args.out, exist_ok=True)
    grid = np.arange(args.wl_min, args.wl_max + 1e-9, args.step)

    zips = download(args.data_dir, args.data)
    Xraw, meta = load_spectra(zips, grid)
    print(f"Loaded {len(Xraw)} spectra from {len({m['sample'] for m in meta})} specimens "
          f"({grid[0]:.0f}-{grid[-1]:.0f} nm, {len(grid)} points)")
    print("Forms:", dict(Counter(m["form"] for m in meta)))

    Xp = preprocess(Xraw, args.deriv, args.window)
    task_classification(Xp, meta, args, grid, Xraw, args.out)
    task_blend(Xp, meta, args, args.out)


if __name__ == "__main__":
    main()
