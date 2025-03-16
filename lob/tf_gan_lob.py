import os
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm.notebook import tqdm

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}")
else:
    print("Using CPU")

@dataclass
class GANConfig:
    batch_size: int = 64
    z_dim: int = 100
    lob_dim: int = 40
    epochs: int = 50
    learning_rate_d: float = 0.0002
    learning_rate_g: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.1
    generator_target_prob: float = 0.65

config = GANConfig()

# Load and preprocess LOB data
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

# Define Generator model with financial constraints
def build_generator():
    inputs = layers.Input(shape=(config.z_dim,))
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(config.lob_dim, activation='tanh')(x)
    # Apply penalties using Lambda layers
    penalties = layers.Lambda(lambda x: tf.nn.softplus(-x))(x)  # Ensure non-negative prices and quantities
    bid_prices = x[:, :10]
    ask_prices = x[:, 20:30]
    bid_diff = bid_prices[:, :-1] - bid_prices[:, 1:]
    ask_diff = ask_prices[:, 1:] - ask_prices[:, :-1]
    # Fix: Padding to maintain shape consistency
    bid_diff_padded = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, 1]]))(bid_diff)
    ask_diff_padded = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 0]]))(ask_diff)
    penalties = layers.Concatenate(axis=1)([penalties, bid_diff_padded, ask_diff_padded])
    max_bid = layers.Lambda(lambda x: tf.reduce_logsumexp(x, axis=1, keepdims=True))(bid_prices)
    max_ask = layers.Lambda(lambda x: tf.reduce_logsumexp(x, axis=1, keepdims=True))(ask_prices)
    penalties += layers.Lambda(lambda x: tf.nn.softplus(x))(max_bid - max_ask)
    model = models.Model(inputs, [x, layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(penalties)])
    return model

# Define Discriminator model with financial constraints
def build_discriminator():
    inputs = layers.Input(shape=(config.lob_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Apply penalties using Lambda layers
    penalties = layers.Lambda(lambda x: tf.nn.softplus(-x))(inputs)
    bid_prices = inputs[:, :10]
    ask_prices = inputs[:, 20:30]
    bid_diff = bid_prices[:, :-1] - bid_prices[:, 1:]
    ask_diff = ask_prices[:, 1:] - ask_prices[:, :-1]
    
    # Fix: Padding to maintain shape consistency
    bid_diff_padded = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, 1]]))(bid_diff)
    ask_diff_padded = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 0]]))(ask_diff)
    penalties = layers.Concatenate(axis=1)([penalties, bid_diff_padded, ask_diff_padded])
    
    max_bid = layers.Lambda(lambda x: tf.reduce_logsumexp(x, axis=1, keepdims=True))(bid_prices)
    max_ask = layers.Lambda(lambda x: tf.reduce_logsumexp(x, axis=1, keepdims=True))(ask_prices)
    penalties += layers.Lambda(lambda x: tf.nn.softplus(x))(max_bid - max_ask)
    
    model = models.Model(inputs, [output, layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(penalties)])
    return model


# Define faulty rate computation
def compute_faulty_rate(lob_tensor):
    bid_prices = lob_tensor[:, :10]
    ask_prices = lob_tensor[:, 20:30]
    bid_quantities = lob_tensor[:, 10:20]
    ask_quantities = lob_tensor[:, 30:40]

    faulty_count = tf.reduce_sum(tf.cast(bid_prices[:, 0] >= ask_prices[:, 0], tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(bid_prices[:, :-1] <= bid_prices[:, 1:], tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(ask_prices[:, :-1] >= ask_prices[:, 1:], tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(bid_quantities < 0, tf.float32))
    faulty_count += tf.reduce_sum(tf.cast(ask_quantities < 0, tf.float32))
    
    total_elements = tf.size(lob_tensor, out_type=tf.float32)
    faulty_rate = faulty_count / total_elements
    return faulty_rate


# Initialize models
generator = build_generator()
discriminator = build_discriminator()

optimizer_g = optimizers.Adam(learning_rate=config.learning_rate_g, beta_1=config.beta1, beta_2=config.beta2)
optimizer_d = optimizers.Adam(learning_rate=config.learning_rate_d, beta_1=config.beta1, beta_2=config.beta2)

bce_loss = tf.keras.losses.BinaryCrossentropy()

lambda_penalty = 2  # Adjust this weight based on importance of penalty term

for epoch in tqdm(range(config.epochs), desc='Training Progress'):
    for real_batch in lob_dataset:
        batch_size = tf.shape(real_batch)[0]
        real_labels = tf.ones((batch_size, 1)) * (1 - config.label_smoothing)
        fake_labels = tf.zeros((batch_size, 1))

        z = tf.random.normal((batch_size, config.z_dim))
        fake_data, fake_penalty = generator(z)

        # Train Discriminator
        with tf.GradientTape() as tape_d:
            real_pred, real_penalty = discriminator(real_batch)
            fake_pred, fake_penalty = discriminator(fake_data)

            # Binary cross-entropy loss
            loss_d = bce_loss(real_labels, real_pred) + bce_loss(fake_labels, fake_pred)

            # Add penalty term to discriminator loss
            loss_d += lambda_penalty * (tf.reduce_mean(real_penalty) + tf.reduce_mean(fake_penalty))

        grads_d = tape_d.gradient(loss_d, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape_g:
            fake_data, fake_penalty = generator(z)
            fake_pred, fake_penalty = discriminator(fake_data)

            # Generator loss (trying to fool the discriminator)
            loss_g = bce_loss(real_labels, fake_pred)

            # Add penalty term to generator loss
            loss_g += lambda_penalty * tf.reduce_mean(fake_penalty)

        grads_g = tape_g.gradient(loss_g, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{config.epochs} - Loss D: {loss_d.numpy():.4f}, Loss G: {loss_g.numpy():.4f}")

# Generate synthetic LOB data
z = tf.random.normal((10, config.z_dim))
synthetic_lob, _ = generator(z)  # Extract only the generated LOB data
synthetic_lob = synthetic_lob.numpy()
synthetic_lob = scaler.inverse_transform(synthetic_lob)
synthetic_lob_df = pd.DataFrame(synthetic_lob, columns=lob_features)


# Compute and print faulty rate
synthetic_lob_tensor = tf.convert_to_tensor(synthetic_lob, dtype=tf.float32)
faulty_rate = compute_faulty_rate(synthetic_lob_tensor)
print("Faulty Rate for Synthetic Data:", faulty_rate.numpy())

# Print synthetic LOB data
print("Synthetic LOB Data:")
print(synthetic_lob_df.head())
