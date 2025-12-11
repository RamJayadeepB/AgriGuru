import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import umap.umap_ as umap
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

print("Starting model training script...")

# ---------------------------
# Base dir & file paths (relative)
# ---------------------------
BASE_DIR = os.path.dirname(__file__) or "."
CSV_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.joblib")

if not os.path.exists(CSV_PATH):
    print(f"ERROR: '{CSV_PATH}' not found.")
    raise SystemExit(1)

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully:", CSV_PATH)

# ---------------------------
# 2. Filter Crops
# ---------------------------
agricultural_crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil'
]
horticultural_crops = [
    'banana', 'mango', 'grapes', 'apple', 'orange', 'papaya','pomegranate',
    'coconut', 'cotton', 'coffee', 'jute', 'watermelon', 'muskmelon'
]
selected = agricultural_crops + horticultural_crops
df_mixed = df[df['label'].isin(selected)].reset_index(drop=True)
print(f"Training on {len(df_mixed)} mixed samples.")

# ---------------------------
# 3. Encode and Train
# ---------------------------
X = df_mixed.drop('label', axis=1).copy()
y = df_mixed['label'].copy()

# Save model columns for later inference (order matters)
model_columns = list(X.columns)
joblib.dump(model_columns, COLUMNS_PATH)
print("Saved model columns:", COLUMNS_PATH)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, ENCODER_PATH)
print("Saved encoder:", ENCODER_PATH)

final_model = RandomForestClassifier(
    max_depth=12, max_features='sqrt',
    min_samples_leaf=1, min_samples_split=2,
    n_estimators=100, random_state=42
)
final_model.fit(X, y_encoded)
joblib.dump(final_model, MODEL_PATH)
print("Model trained and saved to:", MODEL_PATH)

# ======================================================
# SHAP + Visualization + K-Fold Evaluation
# ======================================================
print("\n==============================")
print("MODEL ANALYSIS STARTING")
print("==============================")

explainer = shap.TreeExplainer(final_model)
raw_shap = explainer.shap_values(X)

# ---------------------------
# SHAP Aggregation (handles multiclass)
# ---------------------------
print("\nPerforming SHAP explainability...")

if isinstance(raw_shap, list):
    # multiclass -> list of arrays (one per class)
    print(f"Detected multiclass with {len(raw_shap)} outputs.")
    arrs = []
    for i, sv in enumerate(raw_shap):
        sv = np.array(sv)
        print(f" → Class {i} raw shape: {sv.shape}")
        # If there's an unwanted axis, try to reduce it
        if sv.ndim == 3:
            sv = np.mean(sv, axis=-1)
        arrs.append(sv)
    # stack into (n_samples, n_features, n_classes) then average across classes
    stacked = np.stack(arrs, axis=2)
    shap_agg = np.mean(stacked, axis=2)
else:
    shap_agg = np.array(raw_shap)

# If still 3D, reduce last axis
if shap_agg.ndim == 3:
    shap_agg = np.mean(shap_agg, axis=-1)

print("Aggregated SHAP shape:", shap_agg.shape)
print("X shape:", X.shape)

if shap_agg.shape[1] != X.shape[1]:
    raise ValueError(f"SHAP features {shap_agg.shape[1]} != X features {X.shape[1]}")

# ---------------------------
# Feature Importance
# ---------------------------
mean_abs_by_feature = np.mean(np.abs(shap_agg), axis=0).ravel()
feat_importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": mean_abs_by_feature
}).sort_values("mean_abs_shap", ascending=False)

# ---------------------------
# Global SHAP Plots (save, don't show)
# ---------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_agg, X, show=False)
plt.title("SHAP Summary (Aggregated Across Classes)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "shap_summary.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.barplot(x="mean_abs_shap", y="feature", data=feat_importance, palette="mako")
plt.title("Feature Importance (Mean |SHAP|)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "shap_feature_ranking.png"), dpi=300)
plt.close()

# Dependence plots for top 5 features (save each)
for feat in feat_importance["feature"].head(5):
    try:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_agg, X, show=False, interaction_index=None)
        plt.title(f"Dependence plot: {feat}")
        fn = os.path.join(BASE_DIR, f"shap_dependence_{feat}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create dependence plot for {feat}: {e}")

print("SHAP analysis completed and plots saved.")

# ---------------------------
# t-SNE and UMAP Visualization
# ---------------------------
print("\nGenerating t-SNE and UMAP visualizations...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab20', s=40)
plt.title("t-SNE Visualization of Crops")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "tsne_true.png"), dpi=300)
plt.close()

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='tab20', s=40)
plt.title("UMAP Visualization of Crops")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "umap_true.png"), dpi=300)
plt.close()

print("t-SNE and UMAP visualizations saved.")

# ---------------------------
# Stratified K-Fold Metrics
# ---------------------------
print("\nRunning Stratified K-Fold Evaluation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, precs, recs, f1s = [], [], [], []
cm_total = None

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_test = X.iloc[test_idx]
    y_test = y_encoded[test_idx]
    y_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1m = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1m)
    cm = confusion_matrix(y_test, y_pred)
    cm_total = cm if cm_total is None else cm_total + cm

    print(f"Fold {fold}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1m:.4f}")

print("\nCross-Validation Summary:")
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

plt.figure(figsize=(10, 8))
sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues')
plt.title("Aggregated Confusion Matrix (5-Fold CV)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

print("Stratified K-Fold results saved.")

# ---- Load crop names from encoder ----
class_names = list(encoder.classes_)

# ---- Create a labeled DataFrame for CM ----
cm_df = pd.DataFrame(cm_total, index=class_names, columns=class_names)

# ---- Plot Confusion Matrix with labels ----
plt.figure(figsize=(14, 12))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Number of Predictions'})
plt.title("Aggregated Confusion Matrix (5-Fold CV)")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix_labeled.png"), dpi=300)
plt.close()

# ---------------------------
# Summary
# ---------------------------
print("""
==============================
 INTERPRETATION GUIDE
==============================

 SHAP Summary Plot → Global feature influence.
 Bar Plot → Average absolute SHAP by feature.
 Dependence Plots → How individual features affect model prediction.
 t-SNE / UMAP → Show visual class separation.
 5-Fold Evaluation → Stable performance metrics.

  All plots saved to project root.
""")

print("Trained model saved to:", MODEL_PATH)
print("Available classes:", list(encoder.classes_))
