import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Config:
    z_dim = 100  # Dimension of random noise
    batch_size = 32
    learning_rate_g = 0.0002
    learning_rate_d = 0.0002
    feature_matching_weight = 10.0
    epochs = 10

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
        bid_ask_violation = tf.nn.softplus(max_bid - min_ask)

        return penalties, bid_ask_violation

    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        x = self.dense2(x)
        lob_output = self.output_layer(x)
        penalties, bid_ask_violation = self.apply_penalties(lob_output)
        return lob_output, tf.reduce_sum(penalties, axis=1) + bid_ask_violation

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
        bid_ask_violation = tf.nn.softplus(max_bid - min_ask)

        return penalties, bid_ask_violation


    
    def call(self, inputs, training=True, return_features=False):
        penalties, bid_ask_violation = self.apply_penalties(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        features = self.feature_layer(x)
        output = self.output_layer(features)
        if return_features:
            return output, features, tf.reduce_sum(penalties, axis=1) + bid_ask_violation
        return output

class FeatureMatching:
    """Feature matching loss"""
    def __call__(self, real_features, fake_features):
        real_mean = tf.reduce_mean(real_features, axis=0)
        fake_mean = tf.reduce_mean(fake_features, axis=0)
        return tf.reduce_mean(tf.square(real_mean - fake_mean))

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


gan = SimpleGAN(config)
gan.train(lob_dataset, config.epochs)

noise = tf.random.normal([100, config.z_dim])
generated_samples, _ = gan.generator(noise, training=False)  # Ignore the penalties
generated_samples = generated_samples.numpy()
# Convert back to original scale
generated_samples = scaler.inverse_transform(generated_samples.reshape(100, -1))

# Save to CSV
np.savetxt("generated_lob_data.csv", generated_samples, delimiter=",")
print("Generated 100 samples and saved to 'generated_lob_data.csv'")
