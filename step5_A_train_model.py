import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# DATASET
# =========================
class SliceDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []

        files = sorted([f for f in os.listdir(data_path) if f.endswith(".npy")])
        print(f"Found {len(files)} files")

        for file in files:
            volume = np.load(os.path.join(data_path, file))

            volume = volume / (np.max(volume) + 1e-8)
            volume = volume[:, :, :, 0]  # use 1 channel

            D = volume.shape[0]

            for i in range(1, D - 2):
                prev = volume[i - 1]
                next1 = volume[i + 1]
                next2 = volume[i + 2]
                target = volume[i]

                inp = np.stack([prev, next1, next2], axis=0)
                self.samples.append((inp, target))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y


# =========================
# MODEL
# =========================
class SliceNet(nn.Module):
    def __init__(self):
        super(SliceNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# TRAIN
# =========================
def train_model(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SliceDataset(data_path)

    if len(dataset) == 0:
        raise ValueError(" No training data found")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SliceNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 5

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "ml_model.pth")
    print(" Model saved: ml_model.pth")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_model("processed_data")