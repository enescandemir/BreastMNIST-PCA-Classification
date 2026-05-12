# ─── 1. LIBRARIES ───────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from xgboost import XGBClassifier
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
import warnings
warnings.filterwarnings("ignore")
from medmnist import BreastMNIST


# ─── 2. DATASET ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Loading BreastMNIST...")
print("=" * 60)

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = BreastMNIST(split="train", transform=data_transform, download=True, size=224)
val_dataset   = BreastMNIST(split="val",   transform=data_transform, download=True, size=224)
test_dataset  = BreastMNIST(split="test",  transform=data_transform, download=True, size=224)

full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
full_loader  = DataLoader(full_dataset, batch_size=32, shuffle=False)

print(f"  Total samples : {len(full_dataset)}")


# ─── 3. ResNet-50 FEATURE EXTRACTION ────────────────────────────────────────
print("\n" + "=" * 60)
print("  Extracting Features via ResNet-50 (2048 dimensions)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

def extract_features(loader):
    features_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = resnet(imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy().squeeze())
    return np.vstack(features_list), np.concatenate(labels_list)

X_all, y_all = extract_features(full_loader)
print(f"  Full feature matrix shape : {X_all.shape}")   # (780, 2048)


# ─── 4. 5-FOLD CROSS VALIDATION ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting 5-Fold Stratified Cross Validation...")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "KNN (k=5)"    : KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear)" : SVC(kernel="linear", probability=True, max_iter=1000),
    "XGBoost"      : XGBClassifier(n_estimators=100, max_depth=6,
                                   learning_rate=0.3, use_label_encoder=False,
                                   eval_metric="logloss", random_state=42),
}

metrics = ["acc", "auc", "f1", "precision", "recall"]

cv_results = {
    clf_name: {f"full_{m}": [] for m in metrics} |
              {f"pca_{m}":  [] for m in metrics}
    for clf_name in classifiers
}

last_fold_preds = {
    clf_name: {"full": {"y_true": [], "y_pred": []},
               "pca":  {"y_true": [], "y_pred": []}}
    for clf_name in classifiers
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
    print(f"\n  ── Fold {fold}/5 ──")

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    pca_temp = PCA().fit(X_tr_s)
    n_comp   = np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= 0.90) + 1

    pca = PCA(n_components=n_comp)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)

    print(f"     PCA components: {n_comp}  "
          f"(2048 → {n_comp}, 90% variance retained)")

    for clf_name, clf_template in classifiers.items():

        clf_full = clone(clf_template)
        clf_full.fit(X_tr_s, y_tr)
        y_pred_full = clf_full.predict(X_te_s)
        proba_full  = clf_full.predict_proba(X_te_s)[:, 1]

        cv_results[clf_name]["full_acc"].append(accuracy_score(y_te, y_pred_full))
        cv_results[clf_name]["full_auc"].append(roc_auc_score(y_te, proba_full))
        cv_results[clf_name]["full_f1"].append(f1_score(y_te, y_pred_full))
        cv_results[clf_name]["full_precision"].append(precision_score(y_te, y_pred_full))
        cv_results[clf_name]["full_recall"].append(recall_score(y_te, y_pred_full))

        clf_pca = clone(clf_template)
        clf_pca.fit(X_tr_pca, y_tr)
        y_pred_pca = clf_pca.predict(X_te_pca)
        proba_pca  = clf_pca.predict_proba(X_te_pca)[:, 1]

        cv_results[clf_name]["pca_acc"].append(accuracy_score(y_te, y_pred_pca))
        cv_results[clf_name]["pca_auc"].append(roc_auc_score(y_te, proba_pca))
        cv_results[clf_name]["pca_f1"].append(f1_score(y_te, y_pred_pca))
        cv_results[clf_name]["pca_precision"].append(precision_score(y_te, y_pred_pca))
        cv_results[clf_name]["pca_recall"].append(recall_score(y_te, y_pred_pca))

        if fold == 5:
            last_fold_preds[clf_name]["full"]["y_true"] = y_te
            last_fold_preds[clf_name]["full"]["y_pred"] = y_pred_full
            last_fold_preds[clf_name]["pca"]["y_true"]  = y_te
            last_fold_preds[clf_name]["pca"]["y_pred"]  = y_pred_pca

        print(f"     {clf_name:15s} | "
              f"Full → Acc:{cv_results[clf_name]['full_acc'][-1]:.3f} "
              f"F1:{cv_results[clf_name]['full_f1'][-1]:.3f} | "
              f"PCA  → Acc:{cv_results[clf_name]['pca_acc'][-1]:.3f} "
              f"F1:{cv_results[clf_name]['pca_f1'][-1]:.3f}")


# ─── 5. RESULTS TABLE ───────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("  5-FOLD CROSS VALIDATION RESULTS  (Mean ± Std)")
print("=" * 85)
print(f"  {'Method':<28} {'Acc':>12} {'AUC':>12} {'F1':>12} {'Precision':>12} {'Recall':>12}")
print("  " + "-" * 83)

summary = {}

for clf_name, res in cv_results.items():
    for scenario in ["full", "pca"]:
        label = f"{'Full' if scenario=='full' else 'PCA'} | {clf_name}"
        row = {}
        for m in metrics:
            vals = res[f"{scenario}_{m}"]
            row[m] = (np.mean(vals), np.std(vals))
        summary[label] = row

        print(f"  {label:<28} "
              f"{row['acc'][0]:.3f}±{row['acc'][1]:.3f}  "
              f"{row['auc'][0]:.3f}±{row['auc'][1]:.3f}  "
              f"{row['f1'][0]:.3f}±{row['f1'][1]:.3f}  "
              f"{row['precision'][0]:.3f}±{row['precision'][1]:.3f}  "
              f"{row['recall'][0]:.3f}±{row['recall'][1]:.3f}")


# ─── 6. SCREE PLOT ──────────────────────────────────────────────────────────
train_idx_vis, _ = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_all, y_all))
scaler_vis   = StandardScaler()
X_scaled_vis = scaler_vis.fit_transform(X_all[train_idx_vis])
pca_vis      = PCA().fit(X_scaled_vis)
cumvar       = np.cumsum(pca_vis.explained_variance_ratio_)
n_90 = np.argmax(cumvar >= 0.90) + 1
n_95 = np.argmax(cumvar >= 0.95) + 1

plt.figure(figsize=(10, 5))
plt.plot(range(1, min(201, len(cumvar)+1)), cumvar[:200],
         marker='o', markersize=3, linewidth=1.5, color='steelblue')
plt.axhline(y=0.90, color='red',    linestyle='--', label=f'90% Variance ({n_90} components)')
plt.axhline(y=0.95, color='orange', linestyle='--', label=f'95% Variance ({n_95} components)')
plt.axvline(x=n_90, color='red',    linestyle=':', alpha=0.6)
plt.axvline(x=n_95, color='orange', linestyle=':', alpha=0.6)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=150)
plt.show()
print("\n  Scree plot saved → pca_scree_plot.png")


# ─── 7. COMPARISON CHART ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, metric, title in zip(axes,
                              ["acc", "auc"],
                              ["Accuracy (5-Fold CV)", "AUC (5-Fold CV)"]):
    labels_plot = list(summary.keys())
    vals  = [summary[l][metric][0] for l in labels_plot]
    errs  = [summary[l][metric][1] for l in labels_plot]
    colors = ["steelblue" if "Full" in l else "darkorange" for l in labels_plot]

    bars = ax.bar(range(len(labels_plot)), vals, yerr=errs,
                  capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Score (Mean ± Std)")
    ax.grid(axis="y", alpha=0.3)

    for bar, err in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + err + 0.02,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="steelblue",  label="Full 2048 Features"),
                    Patch(color="darkorange", label="PCA-Reduced Features")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Comparison chart saved → comparison_chart.png")


# ─── 8. CONFUSION MATRIX ────────────────────────────────────────────────────
class_names = ["Benign", "Malignant"]
n_clf = len(classifiers)

fig, axes = plt.subplots(2, n_clf, figsize=(5 * n_clf, 9))

for col, clf_name in enumerate(classifiers.keys()):
    for row, scenario in enumerate(["full", "pca"]):
        ax = axes[row][col]
        yt = last_fold_preds[clf_name][scenario]["y_true"]
        yp = last_fold_preds[clf_name][scenario]["y_pred"]
        cm = confusion_matrix(yt, yp)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax, cbar=False)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        prefix = "Full" if scenario == "full" else "PCA"
        ax.set_title(f"{prefix} | {clf_name}")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Confusion matrices saved → confusion_matrices.png")
print("\n  Pipeline complete!")