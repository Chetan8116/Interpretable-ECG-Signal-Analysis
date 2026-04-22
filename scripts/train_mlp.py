import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from train_mlp_v3 import load_and_extract_all

# Project paths (resolve relative to script location)
project_root = script_dir.parent
data_root = project_root / 'archive' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
csv_path = project_root / 'public' / 'ptbxl_database.csv'

# Load dataset
df = pd.read_csv(csv_path)

# Extract features (using subset for faster experimentation)
X, y, _ = load_and_extract_all(data_root, df, max_records=8000)
print('samples', len(X))

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Compute class weights to handle imbalance
classes = np.unique(y_enc)
cw = compute_class_weight('balanced', classes=classes, y=y_enc)
sw = np.array([cw[c] for c in y_enc])

# Train-test split
X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
    X, y_enc, sw,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# Feature scaling
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Define MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=256,
    max_iter=250,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    learning_rate_init=5e-4,
    verbose=False
)

# Train model
mlp.fit(X_tr_s, y_tr, sample_weight=sw_tr)

# Predict
y_pred = mlp.predict(X_te_s)

# Evaluate performance
print('micro_f1', round(f1_score(y_te, y_pred, average='micro'), 4))
print('macro_f1', round(f1_score(y_te, y_pred, average='macro'), 4))

print(classification_report(
    y_te,
    y_pred,
    target_names=le.classes_,
    digits=3,
    zero_division=0
))