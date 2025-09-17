import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Configuration
# ------------------------------
test_folder = "/Users/mudabbir/Documents/SIH/dataset/test"
train_columns_path = "/Users/mudabbir/Documents/SIH/dataset/train/train_columns.pkl"
model_path = "contrastive_model_ntxent.pth"
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold_percentile = 95
top_n = 20

# ------------------------------
# Load CSV/TXT files
# ------------------------------
def load_files_from_folder(folder_path):
    all_files = []
    for file_name in os.listdir(folder_path):
        if file_name in ["train_logs.csv", "test_logs.csv"]:
            continue
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_name.lower().endswith(".txt"):
            try:
                df = pd.read_csv(file_path, sep="\t")
            except:
                df = pd.read_csv(file_path, sep="\s+")
        else:
            continue
        all_files.append(df)
    if not all_files:
        raise FileNotFoundError(f"No CSV/TXT dataset files found in {folder_path}")
    combined_df = pd.concat(all_files, ignore_index=True)
    print(f"Loaded {len(all_files)} files from {folder_path}, total rows: {len(combined_df)}")
    return combined_df

# ------------------------------
# Preprocess dataframe
# ------------------------------
def preprocess_df(df):
    df = df.copy()
    drop_cols = ['src_ip', 'dst_ip']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.astype('float32')
    assert np.isfinite(df.values).all(), "Data contains NaNs or infinite values!"
    return df

# ------------------------------
# Dataset
# ------------------------------
class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values.astype('float32'))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------------
# Encoder model (same as training)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------
# Load and preprocess test data
# ------------------------------
test_df = load_files_from_folder(test_folder)
test_df = preprocess_df(test_df)

# Align test columns to training columns
with open(train_columns_path, "rb") as f:
    train_columns = pickle.load(f)
for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0.0
for col in test_df.columns:
    if col not in train_columns:
        test_df = test_df.drop(col, axis=1)
test_df = test_df[train_columns]

# ------------------------------
# DataLoader
# ------------------------------
test_dataset = TestDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# Load Encoder model
# ------------------------------
model = Encoder(len(train_columns)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Encoder model loaded from {model_path}")

# ------------------------------
# Generate embeddings and anomaly scores
# ------------------------------
embeddings = []
scores = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        z = model(batch)
        embeddings.append(z.cpu())
        score = torch.norm(z, dim=1)  # anomaly score = embedding norm
        scores.extend(score.cpu().numpy())

embeddings = torch.cat(embeddings).numpy()
scores = np.array(scores)

# ------------------------------
# Compute anomaly threshold
# ------------------------------
threshold = np.percentile(scores, threshold_percentile)
labels = ["Anomalous" if s>threshold else "Normal" for s in scores]
print(f"Anomaly threshold ({threshold_percentile}th percentile): {threshold:.4f}")

# ------------------------------
# Save results
# ------------------------------
output_df = pd.DataFrame(test_df)
output_df["anomaly_score"] = scores
output_df["label"] = labels
save_path = os.path.join(test_folder, "test_logs.csv")
output_df.to_csv(save_path, index=False)
print(f"Test logs saved to {save_path}")

# ------------------------------
# Identify top N anomalies
# ------------------------------
top_indices = np.argsort(scores)[-top_n:]
print(f"Top {top_n} anomalies at indices: {top_indices}")

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(10,5))
plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold_percentile}th pct)')
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
hist_path = os.path.join(test_folder, "anomaly_score_histogram.png")
plt.savefig(hist_path)
plt.close()
print(f"Histogram saved to {hist_path}")

plt.figure(figsize=(12,4))
colors = ['red' if s>threshold else 'green' for s in scores]
plt.scatter(range(len(scores)), scores, c=colors, s=10)
plt.scatter(top_indices, scores[top_indices], c='purple', s=50, label=f'Top {top_n} anomalies')
plt.axhline(threshold, color='blue', linestyle='--', label='Threshold')
plt.title("Anomaly Scores Scatter Plot")
plt.xlabel("Sample Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
scatter_path = os.path.join(test_folder, "anomaly_score_scatter_topN.png")
plt.savefig(scatter_path)
plt.close()
print(f"Scatter plot saved to {scatter_path}")