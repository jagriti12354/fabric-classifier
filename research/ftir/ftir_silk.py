"""
ATR-FTIR proof of concept: silk vs cotton vs polyester (and silk vs everything)
Dataset: "A dataset of infrared (ATR-FTIR) spectra for textile fibres of natural
          and man-made origin" (King's College London; Mendeley Data, CC BY 4.0)
  https://data.mendeley.com/datasets/rx3fjgz96x/3   DOI 10.17632/rx3fjgz96x.3
  Instrument: PerkinElmer Frontier, diamond ATR, 4000-550 cm-1, 4 cm-1
  Cite the dataset DOI + the Data in Brief article (2026).

Tasks
  A. 3-class: silk / cotton / polyester (PET)
  B. Silk vs ALL other fibres (incl. wool, alpaca, nylon = the hard "amide" negatives)
  C. One-class silk model (SIMCA-style): trained on silk ONLY, rejects anything else
  + figures: mean spectra with band assignments, silk-vs-wool-vs-nylon amide zoom

All validation is leave-one-SAMPLE-out (replicate spectra of one sample never
appear in both train and test).

Run:
  pip install numpy scipy scikit-learn matplotlib
  python ftir_silk.py
"""

import argparse
import csv
import io
import os
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ZIP_URL = "https://data.mendeley.com/public-api/zip/rx3fjgz96x/download/3"
CSV_NAME = "03_feature_matrix_baseline_corrected_T.csv"   # ALS baseline-corrected, %T

# Spectral windows used for modelling (cm-1). 2400-1800 is skipped: diamond ATR
# absorbs there and CO2 sits at ~2350, so it carries no fibre information.
WINDOWS = [(3700, 2700), (1800, 600)]

BANDS = {3280: "N-H str (amide A)", 2920: "C-H str", 1715: "C=O ester (PET)",
         1625: "amide I", 1515: "amide II", 1240: "C-O ester / amide III",
         1030: "C-O (cellulose)", 725: "Ar C-H (PET)"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def get_csv_text(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "kcl_atr_ftir.zip")
    if not os.path.exists(zpath):
        print("Downloading", ZIP_URL, "(~25 MB)")
        urllib.request.urlretrieve(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(CSV_NAME))
        return z.read(name).decode("latin-1")


def load(data_dir):
    rows = list(csv.reader(io.StringIO(get_csv_text(data_dir))))
    header, body = rows[0], rows[1:]
    wn = np.array([float(h) for h in header[6:]])
    T = np.array([[float(v) for v in r[6:]] for r in body])
    meta = [{"spec": r[0], "sample": r[1], "origin": r[3], "type": r[4],
             "subtype": r[5].strip().strip('"')} for r in body]
    return wn, T, meta


def to_absorbance(T):
    return -np.log10(np.clip(T, 1e-3, None) / 100.0)


def select_windows(wn, X):
    mask = np.zeros_like(wn, dtype=bool)
    for hi, lo in WINDOWS:
        mask |= (wn <= hi) & (wn >= lo)
    return wn[mask], X[:, mask]


def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def preprocess(A, deriv, window):
    Xp = snv(A)
    if deriv:
        Xp = savgol_filter(Xp, window, 2, deriv=deriv, axis=1)
    return Xp


def fibre_label(m):
    s = m["subtype"]
    if s == "Silk":
        return "silk"
    if s == "Cotton":
        return "cotton"
    if s == "PET":
        return "polyester"
    return None


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class PLSDA:
    def __init__(self, n_components=4):
        self.n = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        self.pls = PLSRegression(n_components=min(self.n, X.shape[0] - 1), scale=False).fit(X, Y)
        return self

    def predict(self, X):
        return self.classes_[self.pls.predict(X).argmax(1)]


def svm():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))


def logo_predict(make_model, X, y, groups):
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def sample_level(y, pred, groups):
    t, p = [], []
    for g in np.unique(groups):
        m = groups == g
        t.append(y[m][0])
        vals, cnt = np.unique(pred[m], return_counts=True)
        p.append(vals[cnt.argmax()])
    return np.array(t), np.array(p)


def report(title, y, pred, groups, names):
    print(f"\n--- {title} ---")
    st, sp = sample_level(y, pred, groups)
    print(f"Spectrum accuracy {accuracy_score(y, pred):.3f} | "
          f"sample accuracy {accuracy_score(st, sp):.3f} ({(st == sp).sum()}/{len(st)} samples)")
    print(classification_report(y, pred, labels=names, digits=3, zero_division=0))
    print("Confusion (rows true, cols pred):", names)
    print(confusion_matrix(y, pred, labels=names))
    wrong = sorted({f"{g}({t}->{p})" for g, t, p in zip(groups, y, pred) if t != p})
    if wrong:
        print("Misclassified spectra (sample: true->pred):", ", ".join(wrong))


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def task_three_class(Xp, meta, args):
    idx = [i for i, m in enumerate(meta) if fibre_label(m)]
    X = Xp[idx]
    y = np.array([fibre_label(meta[i]) for i in idx])
    g = np.array([meta[i]["sample"] for i in idx])
    names = ["silk", "cotton", "polyester"]
    print("\n================ TASK A: silk / cotton / polyester ================")
    for n in names:
        print(f"  {n:<10} {np.sum(y == n):>3} spectra, {len(set(g[y == n]))} samples")
    report(f"PLS-DA ({args.lv} LV), leave-one-sample-out",
           y, logo_predict(lambda: PLSDA(args.lv), X, y, g), g, names)
    report("SVM (RBF), leave-one-sample-out", y, logo_predict(svm, X, y, g), g, names)


def task_silk_vs_all(Xp, meta, args):
    X = Xp
    y = np.array(["silk" if m["subtype"] == "Silk" else "non-silk" for m in meta])
    g = np.array([m["sample"] for m in meta])
    print("\n================ TASK B: silk vs ALL other fibres ================")
    print(f"  silk {np.sum(y == 'silk')} spectra / non-silk {np.sum(y == 'non-silk')} spectra "
          f"({len(set(m['subtype'] for m in meta)) - 1} other subtypes incl. wool, alpaca, nylon)")
    for title, mk in [(f"PLS-DA ({args.lv} LV)", lambda: PLSDA(args.lv)), ("SVM (RBF)", svm)]:
        pred = logo_predict(mk, X, y, g)
        report(title + ", leave-one-sample-out", y, pred, g, ["silk", "non-silk"])
        fp = Counter(meta[i]["subtype"] for i in np.where((pred == "silk") & (y == "non-silk"))[0])
        if fp:
            print("Non-silk spectra wrongly called silk, by subtype:", dict(fp))


def task_one_class(Xp, meta, args, out):
    """SIMCA-style: PCA on silk only; accept if Q-residual below threshold."""
    silk = np.array([m["subtype"] == "Silk" for m in meta])
    g = np.array([m["sample"] for m in meta])
    print("\n================ TASK C: one-class silk model (SIMCA-style) ================")
    k = args.pcs

    def q_resid(pca, X):
        R = X - pca.inverse_transform(pca.transform(X))
        return (R ** 2).sum(1)

    # threshold from leave-one-silk-sample-out residuals
    silk_samples = np.unique(g[silk])
    q_silk_cv = []
    for s in silk_samples:
        tr = silk & (g != s)
        te = silk & (g == s)
        pca = PCA(n_components=min(k, tr.sum() - 1)).fit(Xp[tr])
        q_silk_cv += list(q_resid(pca, Xp[te]))
    q_silk_cv = np.array(q_silk_cv)
    thr = np.percentile(q_silk_cv, 95) * args.q_margin

    pca = PCA(n_components=min(k, silk.sum() - 1)).fit(Xp[silk])
    q_other = q_resid(pca, Xp[~silk])
    print(f"PCs={k}, threshold = 95th pct of CV silk residuals x {args.q_margin}")
    print(f"Silk accepted (CV)      : {np.mean(q_silk_cv <= thr):.3f}  "
          f"({np.sum(q_silk_cv <= thr)}/{len(q_silk_cv)} spectra)")
    print(f"Non-silk rejected       : {np.mean(q_other > thr):.3f}  "
          f"({np.sum(q_other > thr)}/{len(q_other)} spectra)")
    other_sub = np.array([m["subtype"] for m in meta])[~silk]
    acc = Counter(str(s) for s in other_sub[q_other <= thr])
    if acc:
        print("Non-silk ACCEPTED as silk, by subtype:", dict(acc))

    # plot Q residuals per group
    plt = _plt()
    groups = {"silk (CV)": q_silk_cv}
    for name in ["Wool", "Alpaca", "PA 6", "Cotton", "PET"]:
        sel = other_sub == name
        if sel.any():
            groups[name] = q_other[sel]
    rest = ~np.isin(other_sub, ["Wool", "Alpaca", "PA 6", "Cotton", "PET"])
    groups["other"] = q_other[rest]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (n, v) in enumerate(groups.items()):
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)), v, s=14)
    ax.axhline(thr, color="red", ls="--", label="acceptance threshold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylabel("Q residual to silk PCA model (log)")
    ax.set_title("One-class silk model: below line = 'looks like silk'")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out, "ftir_one_class_silk.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mean_spectra(wn, A, meta, out):
    plt = _plt()
    groups = [("Silk", "silk"), ("Cotton", "cotton"), ("PET", "polyester"),
              ("Wool", "wool"), ("PA 6", "nylon 6")]
    sub = np.array([m["subtype"] for m in meta])
    An = A / A.max(1, keepdims=True)   # normalise to max for display

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    off = 0
    for key, lab in groups[:3]:
        axes[0].plot(wn, An[sub == key].mean(0) + off, label=lab)
        off += 1.0
    axes[0].set_title("Mean ATR-FTIR spectra (offset): silk vs cotton vs polyester")
    for b, lab in BANDS.items():
        axes[0].axvline(b, color="grey", ls=":", lw=0.8)
        axes[0].text(b, 3.0, lab, rotation=90,
                     va="top", ha="right", fontsize=7)
    axes[0].set_ylabel("Absorbance (norm.) + offset")
    axes[0].invert_xaxis()
    axes[0].legend(loc="upper left")

    zoom = (wn <= 1800) & (wn >= 1450)
    for key, lab in [groups[0], groups[3], groups[4]]:
        sel = sub == key
        Az = A[sel][:, zoom]
        Az = Az / Az.max(1, keepdims=True)
        axes[1].plot(wn[zoom], Az.mean(0), label=lab)
    for b in (1625, 1650, 1515, 1540):
        axes[1].axvline(b, color="grey", ls=":", lw=0.8)
    axes[1].set_title("Amide region zoom: silk vs wool vs nylon (all have amide bands)")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Absorbance (norm.)")
    axes[1].invert_xaxis()
    axes[1].legend()
    fig.tight_layout()
    p = os.path.join(out, "ftir_mean_spectra.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


def plot_pca(Xp, meta, out):
    plt = _plt()
    sub = np.array([m["subtype"] for m in meta])
    keep = np.isin(sub, ["Silk", "Cotton", "PET", "Wool", "PA 6"])
    S = PCA(n_components=2).fit(Xp[keep]).transform(Xp[keep])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key in ["Silk", "Cotton", "PET", "Wool", "PA 6"]:
        sel = sub[keep] == key
        ax.scatter(S[sel, 0], S[sel, 1], s=18, label=key, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of preprocessed ATR-FTIR spectra")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out, "ftir_pca.png")
    fig.savefig(p, dpi=150)
    print("Saved", p)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ATR-FTIR silk proof of concept")
    ap.add_argument("--data-dir", default="ftir_data")
    ap.add_argument("--out", default="ftir_outputs")
    ap.add_argument("--deriv", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--lv", type=int, default=4, help="PLS-DA latent variables")
    ap.add_argument("--pcs", type=int, default=3, help="PCs in one-class silk model")
    ap.add_argument("--q-margin", type=float, default=1.5,
                    help="multiply 95th-pct silk residual by this for the threshold")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    wn_full, T, meta = load(args.data_dir)
    A_full = to_absorbance(T)
    wn, A = select_windows(wn_full, A_full)
    Xp = preprocess(A, args.deriv, args.window)
    print(f"Loaded {len(meta)} spectra, {len({m['sample'] for m in meta})} samples, "
          f"{len({m['subtype'] for m in meta})} subtypes; {len(wn)} wavenumbers used")

    plot_mean_spectra(wn_full, A_full, meta, args.out)
    plot_pca(Xp, meta, args.out)
    task_three_class(Xp, meta, args)
    task_silk_vs_all(Xp, meta, args)
    task_one_class(Xp, meta, args, args.out)


if __name__ == "__main__":
    main()
