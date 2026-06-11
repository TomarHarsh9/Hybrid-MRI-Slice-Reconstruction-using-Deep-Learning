import numpy as np

# =========================
# LOAD MODEL
# =========================
def load_model():
    import torch
    import torch.nn as nn

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SliceNet().to(device)
    model.load_state_dict(torch.load("ml_model.pth", map_location=device))
    model.eval()

    return model


# =========================
# GENERATE ML VOLUME
# =========================
def generate_ml_volume(model, corrupted, mask):
    import torch

    device = next(model.parameters()).device

    D, H, W, C = corrupted.shape
    ml_volume = [None] * D

    for i in range(1, D - 2):

        if mask[i]:
            continue

        prev = corrupted[i - 1, :, :, 0]
        next1 = corrupted[i + 1, :, :, 0]
        next2 = corrupted[i + 2, :, :, 0]

        inp = np.stack([prev, next1, next2], axis=0)
        inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0, 0]

        ml_volume[i] = pred

    return ml_volume