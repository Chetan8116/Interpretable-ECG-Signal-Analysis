"""
Analyze and visualize Hybrid ResNet1D predictions.

Shows:
  1. Signal branch embeddings (t-SNE visualization)
  2. Feature branch embeddings
  3. Combined predictions
  4. Feature importance from selected features

Usage:
    python scripts/analyze_hybrid_predictions.py
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust import path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_resnet1d_hybrid import (
    HybridResNet1D, load_features, load_signal, parse_superclass,
    CLASSES, N_CLASSES, DEVICE, ROOT, CSV_PATH, ARTIFACTS, PUBLIC
)

warnings.filterwarnings("ignore")

def extract_embeddings(model, loader):
    """Extract signal and feature embeddings before fusion."""
    model.eval()
    
    sig_embeddings = []
    feat_embeddings = []
    labels = []
    
    with torch.no_grad():
        for sig, feat, y in loader:
            sig = sig.to(DEVICE)
            feat = feat.to(DEVICE)
            
            # Signal branch embedding
            x = model.stem(sig)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.gap(x)
            sig_emb = model.signal_embed(x)
            
            # Feature branch embedding
            feat_emb = model.feature_embed(feat)
            
            sig_embeddings.append(sig_emb.cpu().numpy())
            feat_embeddings.append(feat_emb.cpu().numpy())
            labels.extend(y.cpu().tolist())
    
    sig_embeddings = np.vstack(sig_embeddings)
    feat_embeddings = np.vstack(feat_embeddings)
    labels = np.array(labels)
    
    return sig_embeddings, feat_embeddings, labels


def plot_embeddings(embeddings, labels, title, save_path):
    """t-SNE visualization of embeddings."""
    print(f"Computing t-SNE for {title} ...")
    
    # Subsample if too large (t-SNE is slow)
    if len(embeddings) > 2000:
        indices = np.random.choice(len(embeddings), 2000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=colors[i], label=cls, alpha=0.6, s=20)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Class', loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def analyze_feature_importance(model, feature_names, top_k=30):
    """Analyze which features are most important in feature branch."""
    # Get feature MLP weights (first layer)
    first_layer = model.feature_embed[0]  # Linear(n_features → 256)
    weights = first_layer.weight.data.cpu().numpy()  # (256, n_features)
    
    # Importance = mean absolute weight across output neurons
    importance = np.abs(weights).mean(axis=0)
    
    # Sort and get top-k
    top_indices = np.argsort(importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_indices]
    top_scores = importance[top_indices]
    
    return top_names, top_scores


def plot_feature_importance(feature_names, scores, save_path):
    """Bar plot of feature importance."""
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    
    plt.barh(y_pos, scores, color='steelblue', alpha=0.8)
    plt.yticks(y_pos, feature_names, fontsize=9)
    plt.xlabel('Mean Absolute Weight', fontsize=11)
    plt.title('Top 30 Most Important Features (Feature Branch)', 
              fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_comparison(results_path, save_path):
    """Compare confusion matrices between models."""
    with open(results_path) as f:
        results = json.load(f)
    
    cm = np.array(results["confusion_matrix"])
    
    # Normalize by row (true labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[1], cbar_kws={'label': 'Recall'}, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 80)
    print("  Hybrid ResNet1D Prediction Analysis")
    print("=" * 80)
    print()
    
    # Check if model exists
    model_path = ARTIFACTS / "resnet1d_hybrid_best.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train the hybrid model first: python scripts/train_resnet1d_hybrid.py")
        return
    
    print(f"Loading model from {model_path} ...")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    n_features = ckpt["n_features"]
    
    model = HybridResNet1D(n_classes=N_CLASSES, n_features=n_features).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded HybridResNet1D with {n_features} features")
    print()
    
    # Load preprocessors
    preprocessors = ckpt.get("preprocessors", {})
    if not preprocessors:
        print("Warning: No preprocessors found in checkpoint")
        return
    
    imputer = preprocessors["imputer"]
    var_sel = preprocessors["var_sel"]
    mi_sel = preprocessors["mi_sel"]
    scaler = preprocessors["scaler"]
    
    # Load data (validation set only for speed)
    print("Loading validation data ...")
    df = pd.read_csv(CSV_PATH)
    df["superclass"] = df["scp_codes"].apply(parse_superclass)
    df = df.dropna(subset=["superclass"])
    df["label_idx"] = df["superclass"].map({c: i for i, c in enumerate(CLASSES)})
    
    df_val = df[df["strat_fold"] == 9].reset_index(drop=True)
    print(f"  Validation samples: {len(df_val)}")
    
    # Load features
    feat_df = load_features()
    merged = feat_df.merge(df_val[["ecg_id", "superclass", "filename_hr", "label_idx"]], 
                          left_on="record_id", right_on="ecg_id", how="inner")
    
    meta_cols = {"record_id", "ecg_id", "superclass", "strat_fold", "filename_hr", "label_idx"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    X_raw = merged[feat_cols].values.astype(np.float32)
    
    # Preprocess
    X = imputer.transform(X_raw)
    X = var_sel.transform(X)
    X = scaler.transform(X)
    X = mi_sel.transform(X)
    
    # Get selected feature names
    feat_cols_var = [c for c, k in zip(feat_cols, var_sel.get_support()) if k]
    selected_features = [c for c, k in zip(feat_cols_var, mi_sel.get_support()) if k]
    print(f"  Selected features: {len(selected_features)}")
    print()
    
    # Load signals (subsample for speed)
    print("Loading signals (subsampling for speed) ...")
    from torch.utils.data import DataLoader
    from train_resnet1d_hybrid import HybridECGDataset
    
    sigs, labels = [], []
    for i, (_, row) in enumerate(merged.iterrows()):
        if i >= 500:  # Limit for speed
            break
        sig = load_signal(row)
        if sig is not None:
            sigs.append(sig)
            labels.append(int(row["label_idx"]))
    
    X_subset = X[:len(sigs)]
    print(f"  Loaded {len(sigs)} signals")
    print()
    
    # Create dataset and loader
    ds = HybridECGDataset(sigs, X_subset, labels, augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # Extract embeddings
    print("Extracting embeddings ...")
    sig_emb, feat_emb, labels_arr = extract_embeddings(model, loader)
    print(f"  Signal embeddings: {sig_emb.shape}")
    print(f"  Feature embeddings: {feat_emb.shape}")
    print()
    
    # Visualize embeddings
    print("Creating visualizations ...")
    viz_dir = PUBLIC / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    plot_embeddings(sig_emb, labels_arr, 
                   "Signal Branch Embeddings (t-SNE)",
                   viz_dir / "signal_embeddings.png")
    
    plot_embeddings(feat_emb, labels_arr,
                   "Feature Branch Embeddings (t-SNE)", 
                   viz_dir / "feature_embeddings.png")
    
    combined_emb = np.hstack([sig_emb, feat_emb])
    plot_embeddings(combined_emb, labels_arr,
                   "Combined Embeddings (t-SNE)",
                   viz_dir / "combined_embeddings.png")
    
    # Feature importance
    print("Analyzing feature importance ...")
    top_features, top_scores = analyze_feature_importance(model, selected_features)
    plot_feature_importance(top_features, top_scores, 
                           viz_dir / "feature_importance.png")
    
    # Save top features to JSON
    feat_importance = {
        "top_features": [
            {"name": name, "importance": float(score)}
            for name, score in zip(top_features, top_scores)
        ]
    }
    with open(viz_dir / "feature_importance.json", "w") as f:
        json.dump(feat_importance, f, indent=2)
    print(f"  Saved: {viz_dir / 'feature_importance.json'}")
    
    # Confusion matrix
    results_path = PUBLIC / "hybrid_results.json"
    if results_path.exists():
        plot_confusion_comparison(results_path, viz_dir / "confusion_matrix.png")
    
    print()
    print("=" * 80)
    print(f"Analysis complete! Visualizations saved to: {viz_dir}")
    print()
    print("Generated files:")
    print(f"  • signal_embeddings.png     - Signal branch t-SNE")
    print(f"  • feature_embeddings.png    - Feature branch t-SNE")
    print(f"  • combined_embeddings.png   - Fusion t-SNE")
    print(f"  • feature_importance.png    - Top 30 features")
    print(f"  • feature_importance.json   - Feature rankings")
    print(f"  • confusion_matrix.png      - Prediction errors")
    print()
    
    # Print top 10 features
    print("Top 10 Most Important Features:")
    for i, (name, score) in enumerate(zip(top_features[:10], top_scores[:10]), 1):
        print(f"  {i:2d}. {name:<50s}  {score:.4f}")
    print()


if __name__ == "__main__":
    import warnings
    main()
