import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("outputs/compressibility_adv_robustness_2/first_exp/saved_models", exist_ok=True)


# -------------------------
# Model
# -------------------------
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# Frobenius normalization
# -------------------------
def frobenius_normalize(model):
    with torch.no_grad():
        w = model.fc1.weight
        model.fc1.weight /= torch.norm(w, p="fro")


# -------------------------
# Nuclear norm
# -------------------------
def nuclear_norm(model):
    s = torch.linalg.svdvals(model.fc1.weight)
    return s.sum()


# -------------------------
# Accuracy
# -------------------------
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# -------------------------
# Data
# -------------------------
transform = transforms.ToTensor()

train_full = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)

test_set = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

val_size = 5000
train_size = len(train_full) - val_size

train_set, val_set = torch.utils.data.random_split(train_full, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)


# -------------------------
# Experiment parameters
# -------------------------
alphas = [0.0, 5e-3, 1e-2, 5e-2]
seeds = [0, 1, 2]

epochs = 100
patience = 10


# -------------------------
# Training
# -------------------------
for alpha in alphas:

    print("\n====================")
    print("alpha =", alpha)
    print("====================")

    for seed in seeds:

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("\nseed =", seed)

        model = FCN().to(device)

        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):

            model.train()

            for x, y in train_loader:

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                out = model(x)
                ce = criterion(out, y)

                nnr = nuclear_norm(model)

                loss = ce + alpha * nnr

                loss.backward()
                optimizer.step()

                frobenius_normalize(model)

            val_acc = accuracy(model, val_loader)

            print(f"epoch {epoch}  val_acc={val_acc:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("early stopping")
                break

        # load best model
        model.load_state_dict(best_state)

        test_acc = accuracy(model, test_loader)

        print("test accuracy:", test_acc)

        # -------------------------
        # SAVE MODEL
        # -------------------------
        save_path = f"outputs/compressibility_adv_robustness_2/first_exp/saved_models/model_alpha{alpha}_seed{seed}.pt"
        torch.save(model.state_dict(), save_path)

        print("saved:", save_path)