
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import pickle

# ------------------------------
# Configuration
# ------------------------------
train_folder = "/Users/mudabbir/Documents/SIH/dataset/train"
batch_size = 128
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "contrastive_model_ntxent.pth"
temperature = 0.5  # for NT-Xent

# ------------------------------
# Load and preprocess data
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
# Dataset for contrastive pairs
# ------------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values.astype('float32'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Simple augmentation: add small noise as "positive pair"
        x_i = x + 0.01 * torch.randn_like(x)
        x_j = x + 0.01 * torch.randn_like(x)
        return x_i, x_j

# ------------------------------
# Model
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
# NT-Xent Loss
# ------------------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(z, z.T)  # cosine similarity
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=bool)).to(z.device)

    exp_sim = torch.exp(similarity_matrix / temperature) * mask
    log_prob = similarity_matrix / temperature - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    
    positive_idx = torch.arange(batch_size).to(z.device)
    loss = -0.5 * (log_prob[positive_idx, positive_idx + batch_size] + log_prob[positive_idx + batch_size, positive_idx]).mean()
    return loss

# ------------------------------
# Load and preprocess training data
# ------------------------------
train_df = load_files_from_folder(train_folder)
train_df = preprocess_df(train_df)

# Save columns for evaluation
with open(os.path.join(train_folder, "train_columns.pkl"), "wb") as f:
    pickle.dump(train_df.columns.tolist(), f)
print(f"Training columns saved to {os.path.join(train_folder, 'train_columns.pkl')}")

train_dataset = ContrastiveDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# Training
# ------------------------------
model = Encoder(train_df.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_i, x_j in train_loader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        z_i, z_j = model(x_i), model(x_j)
        loss = nt_xent_loss(z_i, z_j, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    train_losses.append(avg_loss)

# ------------------------------
# Save model and logs
# ------------------------------
torch.save(model.state_dict(), model_save_path)
train_logs_path = os.path.join(train_folder, "train_logs.csv")
pd.DataFrame({"epoch": list(range(1, epochs+1)), "avg_loss": train_losses}).to_csv(train_logs_path, index=False)
print(f"Trained model saved as {model_save_path}")
print(f"Training logs saved to {train_logs_path}")