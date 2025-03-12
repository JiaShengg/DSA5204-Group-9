import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

# Hyperparameters
z_dim = 100  # Latent noise vector size
lob_dim = 40  # Adjusted to correct dimension
batch_size = 32  # Batch size for training
epochs = 500  # Number of training epochs
lr = 0.0001  # Learning rate

# Load and preprocess LOB data
file_path = "BTCUSDT-lob.parq"
df = pd.read_parquet(file_path, engine="pyarrow")

# Define LOB features
lob_features = [
    "b0p", "b1p", "b2p", "b3p", "b4p", "b5p", "b6p", "b7p", "b8p", "b9p",
    "b0q", "b1q", "b2q", "b3q", "b4q", "b5q", "b6q", "b7q", "b8q", "b9q",
    "a0p", "a1p", "a2p", "a3p", "a4p", "a5p", "a6p", "a7p", "a8p", "a9p",
    "a0q", "a1q", "a2q", "a3q", "a4q", "a5q", "a6q", "a7q", "a8q", "a9q"
]

# Drop NaN values and sample dataset
df = df.dropna(subset=lob_features).sample(n=5000, random_state=42)

# Normalize data
lob_data = df[lob_features].values
scaler = MinMaxScaler()
lob_data = scaler.fit_transform(lob_data)

# Convert to PyTorch tensor
lob_tensor = torch.tensor(lob_data, dtype=torch.float32)
lob_dataset = TensorDataset(lob_tensor)
lob_loader = DataLoader(lob_dataset, batch_size=batch_size, shuffle=True)

def log_sum_exp(x, dim=1):
    return torch.logsumexp(x, dim=dim) - torch.log(torch.tensor(x.shape[dim], dtype=torch.float32, device=x.device))

# Compute Faulty Rate Function
def compute_faulty_rate(lob_tensor):
    """Calculates the faulty rate for synthetic LOB data using PyTorch tensors."""
    
    bid_prices = lob_tensor[:, :10]  # First 10 columns are bid prices
    ask_prices = lob_tensor[:, 20:30]  # Columns 20-30 are ask prices
    bid_quantities = lob_tensor[:, 10:20]  # Columns 10-20 are bid quantities
    ask_quantities = lob_tensor[:, 30:40]  # Columns 30-40 are ask quantities

    faulty_count = torch.zeros(1, device=lob_tensor.device)  # Initialize faulty count

    # 1. Ensure best bid price < best ask price (b0p < a0p)
    faulty_count += (bid_prices[:, 0] >= ask_prices[:, 0]).sum()

    # 2. Bid prices should be in descending order (b0p > b1p > ...)
    faulty_count += (torch.diff(bid_prices, dim=1) >= 0).sum()

    # 3. Ask prices should be in ascending order (a0p < a1p < ...)
    faulty_count += (torch.diff(ask_prices, dim=1) <= 0).sum()

    # 4. Bid and ask quantities should be non-negative
    faulty_count += (bid_quantities < 0).sum()
    faulty_count += (ask_quantities < 0).sum()
    faulty_count += (bid_prices < 0).sum()
    faulty_count += (ask_prices < 0).sum()

    # Compute faulty rate as a percentage of all elements
    total_elements = lob_tensor.numel()
    faulty_rate = faulty_count / total_elements  # Returns a tensor

    return faulty_rate

# Modify Generator to include Log-Sum-Exp Bid-Ask Spread Penalty
class LOBGenerator(nn.Module):
    def __init__(self, input_dim=100, output_dim=40):
        super(LOBGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim)  # Directly output raw LOB
        )

    def forward(self, z):
        lob_output = self.model(z)  # Unconstrained LOB output

        # **Apply penalties for financial consistency**
        penalties = torch.zeros_like(lob_output)

        # Ensure non-negative prices and quantities
        penalties += F.softplus(-lob_output)

        # Ensure bid prices are descending
        bid_prices = lob_output[:, :10]  
        bid_diff = torch.diff(bid_prices, dim=1)  
        penalties[:, 1:10] += F.softplus(bid_diff)  # Penalize positive differences

        # Ensure ask prices are ascending
        ask_prices = lob_output[:, 20:30]  
        ask_diff = torch.diff(ask_prices, dim=1)
        penalties[:, 21:30] += F.softplus(-ask_diff)  # Penalize negative differences

        # Ensure bid quantities and ask quantities are non-negative
        bid_quantities = lob_output[:, 10:20]
        ask_quantities = lob_output[:, 30:40]
        penalties[:, 10:20] += F.softplus(-bid_quantities)
        penalties[:, 30:40] += F.softplus(-ask_quantities)

        # **New Penalty: Best Bid - Best Ask should be negative**
        max_bid = log_sum_exp(bid_prices)
        max_ask = log_sum_exp(ask_prices)
        bid_ask_violation = F.softplus(max_bid - max_ask)  # Penalizes positive values

        return lob_output, penalties.sum(dim=1) + bid_ask_violation

# Modify Discriminator to include Log-Sum-Exp Bid-Ask Spread Penalty
class EnhancedLOBDiscriminator(nn.Module):
    def __init__(self, input_dim=lob_dim):
        super(EnhancedLOBDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.model(x)
        pred = torch.sigmoid(logits)

        # **Apply penalties for incorrect financial logic**
        penalties = torch.zeros_like(x)

        # Penalize negative prices and quantities
        penalties += F.softplus(-x)

        # Penalize bid prices not descending
        bid_prices = x[:, :10]  
        bid_diff = torch.diff(bid_prices, dim=1)
        penalties[:, 1:10] += F.softplus(bid_diff)

        # Penalize ask prices not ascending
        ask_prices = x[:, 20:30]  
        ask_diff = torch.diff(ask_prices, dim=1)
        penalties[:, 21:30] += F.softplus(-ask_diff)

        # Penalize negative bid and ask quantities
        penalties[:, 10:20] += F.softplus(-x[:, 10:20])
        penalties[:, 30:40] += F.softplus(-x[:, 30:40])

        # **New Penalty: Best Bid - Best Ask should be negative**
        max_bid = log_sum_exp(bid_prices)
        max_ask = log_sum_exp(ask_prices)
        bid_ask_violation = F.softplus(max_bid - max_ask)  # Penalizes positive values

        return pred, penalties.sum(dim=1) + bid_ask_violation

# Initialize GAN Components
generator = LOBGenerator()
discriminator = EnhancedLOBDiscriminator()
optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Train GAN
for epoch in range(epochs):
    for real_batch in lob_loader:
        real_data = real_batch[0].view(-1, lob_dim)
        batch_size = real_data.shape[0]
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Generate fake LOB data
        z = torch.randn(batch_size, z_dim)
        fake_data, penalty_G = generator(z)
        fake_data = fake_data.detach()

        # Discriminator Training
        optim_D.zero_grad()
        real_pred, penalty_D_real = discriminator(real_data)
        fake_pred, penalty_D_fake = discriminator(fake_data)

        loss_D_real = criterion(real_pred, real_labels)  
        loss_D_fake = criterion(fake_pred, fake_labels)  
        loss_D_penalty = penalty_D_real.mean() + penalty_D_fake.mean()  # Discriminator Penalty
        loss_D = loss_D_real + loss_D_fake + 0.1 * loss_D_penalty  # Weighted penalty

        loss_D.backward()
        optim_D.step()

        # Generator Training
        optim_G.zero_grad()
        fake_data, penalty_G = generator(z)
        fake_pred, _ = discriminator(fake_data)

        loss_G_fake = criterion(fake_pred, real_labels)  
        loss_G = loss_G_fake + 0.01 * penalty_G.mean()  # Weighted generator penalty

        loss_G.backward()
        optim_G.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss D = {loss_D.item():.4f}, Loss G = {loss_G.item():.4f}")

# Generate Synthetic LOB Data
z = torch.randn(10, z_dim)
synthetic_lob, _ = generator(z)  # Ignore penalties during inference
synthetic_lob = synthetic_lob.detach().numpy()
synthetic_lob = scaler.inverse_transform(synthetic_lob)  # Reverse normalization

# Convert to PyTorch tensor for faulty rate calculation
synthetic_lob_tensor = torch.tensor(synthetic_lob, dtype=torch.float32)

# Compute Faulty Rate
faulty_rate = compute_faulty_rate(synthetic_lob_tensor)  # Apply only to generated data
print("Faulty Rate for Synthetic Data:", faulty_rate.item())

# Convert to DataFrame for Inspection
synthetic_lob_df = pd.DataFrame(synthetic_lob, columns=lob_features)
print("Synthetic LOB Data:")
print(synthetic_lob_df.head())

