"""
CSCE 636 - Project 1: Deep Neural Network for Estimating m-Height

Note: Claude.ai was provided my notes from Dr. Jiang's research paper
*Analog Error-Correcting Codes* and my LP correctness verification results.
This script is intended for personal reference while working on Project 1.

Overview:
    Train a DNN to estimate the m-height of an Analog Code given its
    systematic generator matrix G = [I_k | P] and parameters n, k, m.

    Cost function: delta(y, y_hat) = (log2(y) - log2(y_hat))^2

    Parameter space: n=9, k in {4,5,6}, m in {2,...,n-k}
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================================================
# CONFIG - update paths as needed
# ==============================================================
TRAIN_DATA_PATH   = 'CSCE-636-Project-1-Train-n_k_m_P'
TRAIN_HEIGHT_PATH = 'CSCE-636-Project-1-Train-mHeights'
MODEL_SAVE_PATH   = 'best_model.pth'
NUM_EPOCHS        = 100
BATCH_SIZE        = 256
LEARNING_RATE     = 1e-3

# ==============================================================
# DEVICE
# ==============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==============================================================
# LOAD DATA
# ==============================================================
with open(TRAIN_DATA_PATH, 'rb') as f:
    train_data = pickle.load(f)
with open(TRAIN_HEIGHT_PATH, 'rb') as f:
    train_heights = pickle.load(f)

print(f'Total samples: {len(train_data)}')

# ==============================================================
# FEATURE ENCODING
#
# P matrices have variable shapes (k varies 4-6, n-k varies 3-5).
# Strategy: flatten P, zero-pad to max size 6x5=30, normalize to [-1,1].
# Prepend normalized n, k, m. Final vector: 33 features.
# Target: predict log2(m-height) to stabilize training and match loss function.
# ==============================================================
def encode_sample(sample):
    n, k, m, P = sample
    p_flat = P.flatten()
    p_padded = np.zeros(30)
    p_padded[:len(p_flat)] = p_flat
    p_padded = p_padded / 100.0  # normalize to [-1, 1]
    features = np.concatenate([[n / 9.0, k / 9.0, m / 9.0], p_padded])
    return features.astype(np.float32)

X = np.array([encode_sample(s) for s in train_data])
y = np.array([np.log2(h) for h in train_heights], dtype=np.float32)

print(f'Feature shape: {X.shape}')
print(f'Target (log2 height) range: {y.min():.2f} to {y.max():.2f}')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f'Train: {len(X_train)}, Val: {len(X_val)}')

# ==============================================================
# DATASET
# ==============================================================
class MHeightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(MHeightDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(MHeightDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================
# MODEL
#
# Fully connected DNN: 33 -> 256 -> 512 -> 256 -> 128 -> 1
# BatchNorm + Dropout for regularization.
# Output is a single scalar (log2 m-height prediction).
# ==============================================================
class MHeightDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(33, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MHeightDNN().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params:,}')

# ==============================================================
# LOSS, OPTIMIZER, SCHEDULER
#
# Since we predict log2(y_hat), the cost function simplifies to MSE:
#   delta = (log2(y) - log2(y_hat))^2 = (y_true - y_pred)^2
# ==============================================================
def custom_loss(pred, true):
    return torch.mean((true - pred) ** 2)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ==============================================================
# TRAINING LOOP
# ==============================================================
best_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = custom_loss(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += custom_loss(model(X_batch), y_batch).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        print(f'Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}')

print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')

# ==============================================================
# PLOT TRAINING CURVES
# ==============================================================
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log2 space)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ==============================================================
# EVALUATION - per (n,k,m) breakdown on validation set
# ==============================================================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        all_preds.extend(model(X_batch.to(device)).cpu().numpy())
        all_true.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

print(f'\nOverall average cost (delta): {np.mean((all_true - all_preds)**2):.4f}')

_, val_idx = train_test_split(np.arange(len(train_data)), test_size=0.1, random_state=42)
val_nkm = [(train_data[i][0], train_data[i][1], train_data[i][2]) for i in val_idx]

print('\nPer (n,k,m) average cost:')
for key in sorted(set(val_nkm)):
    mask = [i for i, v in enumerate(val_nkm) if v == key]
    cost = np.mean((all_true[mask] - all_preds[mask]) ** 2)
    print(f'  n={key[0]}, k={key[1]}, m={key[2]}: {cost:.4f}')

# ==============================================================
# INFERENCE FUNCTION - use this to predict on new data
# ==============================================================
def predict(test_data, model_path=MODEL_SAVE_PATH):
    """
    Predict m-heights for a list of (n, k, m, P) samples.

    Args:
        test_data: list of [n, k, m, P] samples
        model_path: path to saved model weights

    Returns:
        numpy array of predicted m-heights (>= 1.0)
    """
    model = MHeightDNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_test = torch.tensor(
        np.array([encode_sample(s) for s in test_data]),
        dtype=torch.float32
    )

    with torch.no_grad():
        log2_preds = model(X_test.to(device)).cpu().numpy()

    # Convert from log2 space, clamp to >= 1 as required
    return np.maximum(2.0 ** log2_preds, 1.0)
