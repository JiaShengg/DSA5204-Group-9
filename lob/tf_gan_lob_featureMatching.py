import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ==========================================================================================
# config and data
# ==========================================================================================


class Config:
    z_dim = 40*100  # Dimension of random noise
    batch_size = 32
    learning_rate_g = 0.0002
    learning_rate_d = 0.0002
    feature_matching_weight = 1
    epochs = 20
    penalty_weight = 5

config = Config()

file_path = "BTCUSDT-lob.parq"
df = pd.read_parquet(file_path, engine="pyarrow")

lob_features = [
    "b0p", "b1p", "b2p", "b3p", "b4p", "b5p", "b6p", "b7p", "b8p", "b9p",
    "b0q", "b1q", "b2q", "b3q", "b4q", "b5q", "b6q", "b7q", "b8q", "b9q",
    "a0p", "a1p", "a2p", "a3p", "a4p", "a5p", "a6p", "a7p", "a8p", "a9p",
    "a0q", "a1q", "a2q", "a3q", "a4q", "a5q", "a6q", "a7q", "a8q", "a9q"
]

df = df.dropna(subset=lob_features).sample(n=5000, random_state=42)
scaler = MinMaxScaler()
lob_data = scaler.fit_transform(df[lob_features].values).astype(np.float32)

lob_dataset = tf.data.Dataset.from_tensor_slices(lob_data).batch(config.batch_size)

# ==========================================================================================
# generator
# ==========================================================================================

class Generator(models.Model):
    """Simple Generator that transforms noise into a 40-dimensional feature vector with financial constraints"""
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.z_dim = config.z_dim
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(40, activation='tanh')
    
    def apply_penalties(self, lob_output):
        penalties = tf.zeros_like(lob_output)

        # Ensure non-negative prices and quantities
        penalties += tf.nn.softplus(-lob_output)

        # Ensure bid prices are descending
        bid_prices = lob_output[:, :10]
        bid_diff = bid_prices[:, 1:] - bid_prices[:, :-1]
        bid_penalties = tf.nn.softplus(bid_diff)

        # Ensure ask prices are ascending
        ask_prices = lob_output[:, 20:30]
        ask_diff = ask_prices[:, 1:] - ask_prices[:, :-1]
        ask_penalties = tf.nn.softplus(-ask_diff)

        # Ensure bid and ask quantities are non-negative
        batch_size = tf.shape(lob_output)[0]

        # Create index tensors for updates
        bid_q_indices = tf.range(10, 20)  # Shape (10,)
        ask_q_indices = tf.range(30, 40)  # Shape (10,)

        # Expand batch dimension
        batch_indices = tf.range(batch_size)[:, None]  # Shape (batch_size, 1)

        bid_q_indices = tf.tile(bid_q_indices[None, :], [batch_size, 1])  # Shape (batch_size, 10)
        ask_q_indices = tf.tile(ask_q_indices[None, :], [batch_size, 1])  # Shape (batch_size, 10)

        bid_q_indices = tf.stack([tf.tile(batch_indices, [1, 10]), bid_q_indices], axis=-1)
        ask_q_indices = tf.stack([tf.tile(batch_indices, [1, 10]), ask_q_indices], axis=-1)

        bid_q_indices = tf.reshape(bid_q_indices, [-1, 2])
        ask_q_indices = tf.reshape(ask_q_indices, [-1, 2])

        bid_q_penalties = tf.reshape(tf.nn.softplus(-lob_output[:, 10:20]), [-1])
        ask_q_penalties = tf.reshape(tf.nn.softplus(-lob_output[:, 30:40]), [-1])

        penalties = tf.tensor_scatter_nd_add(penalties, bid_q_indices, bid_q_penalties)
        penalties = tf.tensor_scatter_nd_add(penalties, ask_q_indices, ask_q_penalties)

        # Ensure Best Bid - Best Ask is negative
        max_bid = tf.math.reduce_logsumexp(bid_prices, axis=1, keepdims=True)
        min_ask = -tf.math.reduce_logsumexp(-ask_prices, axis=1, keepdims=True)
        bid_ask_violation = tf.nn.softplus(min_ask - max_bid)

        return config.penalty_weight*penalties, bid_ask_violation + tf.reduce_sum(bid_penalties) + tf.reduce_sum(ask_penalties)

    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        x = self.dense2(x)
        lob_output = self.output_layer(x)
        penalties, bid_ask_violation = self.apply_penalties(lob_output)
        return lob_output, tf.reduce_sum(penalties, axis=1) + bid_ask_violation

# ==========================================================================================
# discriminator
# ==========================================================================================

class Discriminator(models.Model):
    """Simple Discriminator for classifying real vs fake feature vectors with financial constraints"""
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        
        self.dense1 = layers.Dense(128, activation='relu', input_shape=(40,))
        self.dense2 = layers.Dense(64, activation='relu')
        self.feature_layer = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def apply_penalties(self, lob_output):
        penalties = tf.zeros_like(lob_output)

        # Ensure non-negative prices and quantities
        penalties += tf.nn.softplus(-lob_output)

        # Ensure bid prices are descending
        bid_prices = lob_output[:, :10]
        bid_diff = bid_prices[:, 1:] - bid_prices[:, :-1]
        bid_penalties = tf.nn.softplus(bid_diff)

        # Ensure ask prices are ascending
        ask_prices = lob_output[:, 20:30]
        ask_diff = ask_prices[:, 1:] - ask_prices[:, :-1]
        ask_penalties = tf.nn.softplus(-ask_diff)

        # Ensure bid and ask quantities are non-negative
        batch_size = tf.shape(lob_output)[0]

        # Create index tensors for updates
        bid_q_indices = tf.range(10, 20)  # Shape (10,)
        ask_q_indices = tf.range(30, 40)  # Shape (10,)

        # Expand batch dimension
        batch_indices = tf.range(batch_size)[:, None]  # Shape (batch_size, 1)

        bid_q_indices = tf.tile(bid_q_indices[None, :], [batch_size, 1])  # Shape (batch_size, 10)
        ask_q_indices = tf.tile(ask_q_indices[None, :], [batch_size, 1])  # Shape (batch_size, 10)

        bid_q_indices = tf.stack([tf.tile(batch_indices, [1, 10]), bid_q_indices], axis=-1)
        ask_q_indices = tf.stack([tf.tile(batch_indices, [1, 10]), ask_q_indices], axis=-1)

        bid_q_indices = tf.reshape(bid_q_indices, [-1, 2])
        ask_q_indices = tf.reshape(ask_q_indices, [-1, 2])

        bid_q_penalties = tf.reshape(tf.nn.softplus(-lob_output[:, 10:20]), [-1])
        ask_q_penalties = tf.reshape(tf.nn.softplus(-lob_output[:, 30:40]), [-1])

        penalties = tf.tensor_scatter_nd_add(penalties, bid_q_indices, bid_q_penalties)
        penalties = tf.tensor_scatter_nd_add(penalties, ask_q_indices, ask_q_penalties)

        # Ensure Best Bid - Best Ask is negative
        max_bid = tf.math.reduce_logsumexp(bid_prices, axis=1, keepdims=True)
        min_ask = -tf.math.reduce_logsumexp(-ask_prices, axis=1, keepdims=True)
        bid_ask_violation = tf.nn.softplus(min_ask - max_bid)

        return config.penalty_weight*penalties, bid_ask_violation + tf.reduce_sum(bid_penalties) + tf.reduce_sum(ask_penalties)
    
    def call(self, inputs, training=True, return_features=False):
        penalties, bid_ask_violation = self.apply_penalties(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        features = self.feature_layer(x)
        output = self.output_layer(features)
        if return_features:
            return output, features, tf.reduce_sum(penalties, axis=1) + bid_ask_violation
        return output

# ==========================================================================================
# feature matching
# ==========================================================================================

class FeatureMatching:
    """Feature matching loss combining full feature comparison and bid-offer spread"""
    def __call__(self, real_features, fake_features):
        # Compute overall mean feature matching
        real_mean = tf.reduce_mean(real_features, axis=0)
        fake_mean = tf.reduce_mean(fake_features, axis=0)
        full_feature_loss = tf.reduce_mean(tf.square(real_mean - fake_mean))
        
        # Compute bid-offer spread for real and fake data
        real_bid_prices = real_features[:, :10]  # First 10 columns are bid prices
        real_ask_prices = real_features[:, 20:30]  # Columns 20-30 are ask prices
        real_spread = real_ask_prices - real_bid_prices  # Spread at each level
        real_spread_mean = tf.reduce_mean(real_spread, axis=0)  # Mean across batch

        fake_bid_prices = fake_features[:, :10]
        fake_ask_prices = fake_features[:, 20:30]
        fake_spread = fake_ask_prices - fake_bid_prices
        fake_spread_mean = tf.reduce_mean(fake_spread, axis=0)

        # Compute feature matching loss based on spread difference
        spread_feature_loss = tf.reduce_mean(tf.square(real_spread_mean - fake_spread_mean))
        
        # Combine both losses
        return full_feature_loss + spread_feature_loss

# ==========================================================================================
# GAN
# ==========================================================================================

class SimpleGAN:
    """Simplest GAN with Feature Matching and Financial Constraints"""
    def __init__(self, config):
        self.config = config
        
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.feature_matching = FeatureMatching()
        
        self.gen_optimizer = tf.keras.optimizers.Adam(config.learning_rate_g)
        self.disc_optimizer = tf.keras.optimizers.Adam(config.learning_rate_d)
        
        self.fixed_noise = tf.random.normal([16, config.z_dim])

    def generator_loss(self, fake_output, fake_penalties, real_features, fake_features):
        target = tf.ones_like(fake_output)
        adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, fake_output, from_logits=True))
        fm_loss = self.feature_matching(real_features, fake_features)
        return adv_loss + self.config.feature_matching_weight * fm_loss + tf.reduce_mean(fake_penalties)

    def discriminator_loss(self, real_output, fake_output):
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_output, from_logits=True))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, fake_output, from_logits=True))
        return real_loss + fake_loss
    
    @tf.function
    def train_step(self, real_matrices):
        batch_size = tf.shape(real_matrices)[0]
        noise = tf.random.normal([batch_size, self.config.z_dim])
        
        with tf.GradientTape() as disc_tape:
            fake_matrices, fake_penalties = self.generator(noise, training=True)
            real_output, real_features, _ = self.discriminator(real_matrices, training=True, return_features=True)
            fake_output, fake_features, _ = self.discriminator(fake_matrices, training=True, return_features=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        with tf.GradientTape() as gen_tape:
            fake_matrices, fake_penalties = self.generator(noise, training=True)
            fake_output, fake_features, _ = self.discriminator(fake_matrices, training=True, return_features=True)
            gen_loss = self.generator_loss(fake_output, fake_penalties, real_features, fake_features)
        
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for batch in dataset:
                metrics = self.train_step(batch)
            print(f"Epoch {epoch+1}/{epochs} - Generator Loss: {metrics['gen_loss']:.4f}, Discriminator Loss: {metrics['disc_loss']:.4f}")

# ==========================================================================================
# reasonability check functions
# ==========================================================================================

def compute_faulty_rate(lob_tensor):
    """Calculates the faulty rate for synthetic LOB data using TensorFlow tensors."""
    
    bid_prices = lob_tensor[:, :10]  # First 10 columns are bid prices
    ask_prices = lob_tensor[:, 20:30]  # Columns 20-30 are ask prices
    bid_quantities = lob_tensor[:, 10:20]  # Columns 10-20 are bid quantities
    ask_quantities = lob_tensor[:, 30:40]  # Columns 30-40 are ask quantities

    faulty_count = tf.zeros(1, dtype=tf.float32)  # Initialize faulty count

    # 1. Ensure best bid price < best ask price (b0p < a0p)
    max_bid = tf.reduce_max(bid_prices, axis=1)  # Get the highest bid
    min_ask = tf.reduce_min(ask_prices, axis=1)  # Get the lowest ask
    faulty_count += tf.reduce_sum(tf.cast(max_bid >= min_ask, tf.float32))  # Count violations

    # 2. Bid prices should be in descending order (b0p > b1p > ...)
    faulty_count += tf.reduce_sum(tf.cast(tf.experimental.numpy.diff(bid_prices, axis=1) >= 0, tf.float32))

    # 3. Ask prices should be in ascending order (a0p < a1p < ...)
    faulty_count += tf.reduce_sum(tf.cast(tf.experimental.numpy.diff(ask_prices, axis=1) <= 0, tf.float32))

    # 4. Bid and ask quantities should be non-negative
    faulty_count += tf.reduce_sum(tf.cast(bid_quantities < 0, tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(ask_quantities < 0, tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(bid_prices < 0, tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(ask_prices < 0, tf.float32))

    # Compute faulty rate as a percentage of all elements
    total_elements = tf.size(lob_tensor, out_type=tf.float32)
    faulty_rate = faulty_count / total_elements  # Returns a TensorFlow tensor

    return faulty_rate.numpy()  # Convert to NumPy for further processing

def plot_order_book_scatter(lob_sample):
    """Scatter plot of a single LOB snapshot with price vs. quantity and level labels."""

    # Extract bid and ask prices and sizes
    bid_prices = np.array(lob_sample[:10])
    bid_sizes = np.array(lob_sample[10:20])
    ask_prices = np.array(lob_sample[20:30])
    ask_sizes = np.array(lob_sample[30:40])

    # Remove negative or zero values to ensure valid plotting
    valid_bids = bid_sizes > 0
    valid_asks = ask_sizes > 0
    bid_prices, bid_sizes = bid_prices[valid_bids], bid_sizes[valid_bids]
    ask_prices, ask_sizes = ask_prices[valid_asks], ask_sizes[valid_asks]

    plt.figure(figsize=(8, 5))

    # Scatter plot for bids and asks
    plt.scatter(bid_sizes, bid_prices, color='blue', label="Bids", alpha=0.6)
    plt.scatter(ask_sizes, ask_prices, color='red', label="Asks", alpha=0.6)

    # Annotate each price level with its index
    for i, (q, p) in enumerate(zip(bid_sizes, bid_prices)):
        plt.annotate(f"b{i}", (q, p), fontsize=10, color='blue', ha="right")
    
    for i, (q, p) in enumerate(zip(ask_sizes, ask_prices)):
        plt.annotate(f"a{i}", (q, p), fontsize=10, color='red', ha="left")

    # Adjust x-axis and y-axis dynamically
    max_quantity = max(np.max(bid_sizes) if len(bid_sizes) > 0 else 0,
                       np.max(ask_sizes) if len(ask_sizes) > 0 else 0) * 1.1
    min_price = min(np.min(bid_prices) if len(bid_prices) > 0 else np.inf,
                    np.min(ask_prices) if len(ask_prices) > 0 else np.inf) /1.1
    max_price = max(np.max(bid_prices) if len(bid_prices) > 0 else -np.inf,
                    np.max(ask_prices) if len(ask_prices) > 0 else -np.inf) *1.1

    plt.xlim(0, max_quantity)
    plt.ylim(min_price, max_price)

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title("LOB Scatter Plot (Price vs. Quantity)")

    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================================================================
# train and generate
# ==========================================================================================

gan = SimpleGAN(config)
gan.train(lob_dataset, config.epochs)

noise = tf.random.normal([100, config.z_dim])
generated_samples, _ = gan.generator(noise, training=False)  # Ignore the penalties
generated_samples = generated_samples.numpy()
# Convert back to original scale
generated_samples = scaler.inverse_transform(generated_samples.reshape(100, -1))

# ==========================================================================================
# reasonability check
# ==========================================================================================

faulty_rates = np.array([compute_faulty_rate(tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)) for sample in generated_samples])

# Plot histogram of faulty rates
plt.figure(figsize=(8, 5))
plt.hist(faulty_rates, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Faulty Rate")
plt.ylabel("Frequency")
plt.title("Histogram of Faulty Rates in Generated LOB Data")
plt.grid(True)
plt.show()

# Save to CSV
single_lob_sample = generated_samples[0]  # Choose the first generated sample

# Define column names for LOB structure
lob_columns = [
    "b0p", "b1p", "b2p", "b3p", "b4p", "b5p", "b6p", "b7p", "b8p", "b9p",
    "b0q", "b1q", "b2q", "b3q", "b4q", "b5q", "b6q", "b7q", "b8q", "b9q",
    "a0p", "a1p", "a2p", "a3p", "a4p", "a5p", "a6p", "a7p", "a8p", "a9p",
    "a0q", "a1q", "a2q", "a3q", "a4q", "a5q", "a6q", "a7q", "a8q", "a9q"
]

# Convert to DataFrame
df_single_lob = pd.DataFrame([single_lob_sample], columns=lob_columns)

# Save as CSV
df_single_lob.to_csv("single_generated_lob.csv", index=False)
print("first sample saved to 'single_generated_lob.csv'")

for i in range(5):
    plot_order_book_scatter(generated_samples[i])
