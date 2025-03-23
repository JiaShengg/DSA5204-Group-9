import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import random

# ==========================================================================================
# config and data
# ==========================================================================================

class Config:
    z_dim = 40 * 100
    batch_size = 32
    learning_rate_g = 0.0002
    learning_rate_d = 0.0002
    adv_weight = 1
    epochs = 20
    label_smoothing = 0.1
    lambda_order = 4

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
    def __init__(self, config):
        super(Generator, self).__init__()
        self.dense1 = layers.Dense(512)
        self.bn1 = layers.BatchNormalization()
        self.leaky_relu1 = layers.LeakyReLU(0.2)

        self.dense2 = layers.Dense(256)
        self.bn2 = layers.BatchNormalization()
        self.leaky_relu2 = layers.LeakyReLU(0.2)

        self.dense3 = layers.Dense(128)
        self.bn3 = layers.BatchNormalization()
        self.leaky_relu3 = layers.LeakyReLU(0.2)

        self.dense4 = layers.Dense(64)
        self.bn4 = layers.BatchNormalization()
        self.leaky_relu4 = layers.LeakyReLU(0.2)

        self.output_layer = layers.Dense(40, activation='tanh')

    def call(self, inputs, training=True):
        x = self.leaky_relu1(self.bn1(self.dense1(inputs), training=training))
        x = self.leaky_relu2(self.bn2(self.dense2(x), training=training))
        x = self.leaky_relu3(self.bn3(self.dense3(x), training=training))
        x = self.leaky_relu4(self.bn4(self.dense4(x), training=training))
        return self.output_layer(x)

# ==========================================================================================
# discriminator
# ==========================================================================================

class MinibatchDiscrimination(layers.Layer):
    def __init__(self, num_kernels=50, kernel_dim=5):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        self.T = self.add_weight(
            name="T",
            shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        M = tf.matmul(x, self.T)
        M = tf.reshape(M, [-1, self.num_kernels, self.kernel_dim])
        diffs = tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
        return tf.concat([x, minibatch_features], axis=1)

class Discriminator(models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = layers.Dense(512)
        self.leaky_relu1 = layers.LeakyReLU(0.2)

        self.dense2 = layers.Dense(256)
        self.leaky_relu2 = layers.LeakyReLU(0.2)

        self.dense3 = layers.Dense(128)
        self.leaky_relu3 = layers.LeakyReLU(0.2)

        self.dense4 = layers.Dense(64)
        self.leaky_relu4 = layers.LeakyReLU(0.2)

        self.feature_layer = layers.Dense(32)
        self.leaky_relu5 = layers.LeakyReLU(0.2)

        self.minibatch = MinibatchDiscrimination()
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, return_features=False):
        x = self.leaky_relu1(self.dense1(inputs))
        x = self.leaky_relu2(self.dense2(x))
        x = self.leaky_relu3(self.dense3(x))
        x = self.leaky_relu4(self.dense4(x))
        features = self.leaky_relu5(self.feature_layer(x))
        x = self.minibatch(features)
        output = self.output_layer(x)
        return (output, features) if return_features else output

# ==========================================================================================
# feature matching
# ==========================================================================================

# class FeatureMatching:
#     def __init__(self, sharpness=10.0):
#         self.sharpness = sharpness

#     def __call__(self, real_features, fake_features):
#         def prob_less_matrix(x):
#             x_i = tf.expand_dims(x, axis=2)
#             x_j = tf.expand_dims(x, axis=1)
#             prob_matrix = tf.sigmoid(self.sharpness * (x_j - x_i))
#             return tf.reduce_mean(tf.exp(prob_matrix), axis=0)

#         real_matrix = prob_less_matrix(real_features)
#         fake_matrix = prob_less_matrix(fake_features)

#         match_loss = tf.reduce_mean((tf.square(real_matrix - fake_matrix)))
#         return match_loss

class FeatureMatching:
    def __call__(self, real_features, fake_features):
        # L2 normalize each feature vector (row-wise)
        real_norm = tf.math.l2_normalize(real_features, axis=1)  # (batch, dim)
        fake_norm = tf.math.l2_normalize(fake_features, axis=1)  # (batch, dim)

        # Compute squared pairwise differences: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i • y_j
        # Since x_i and y_j are normalized, their norms are 1, so ||x_i - y_j||^2 = 2 - 2 * (x_i • y_j)
        dot_products = tf.matmul(real_norm, fake_norm, transpose_b=True)  # (batch, batch)
        sq_distances = 2.0 - 2.0 * dot_products  # element-wise squared distance matrix

        # Take the mean over all (i, j)
        return tf.reduce_mean(sq_distances)




# ==========================================================================================
# GAN
# ==========================================================================================

class SimpleGAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator()
        self.feature_matching = FeatureMatching()
        self.gen_optimizer = tf.keras.optimizers.Adam(config.learning_rate_g, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(config.learning_rate_d, beta_1=0.5)
        self.global_step = 0

    def generator_loss(self, fake_output, real_features, fake_features):
        bce = tf.keras.losses.BinaryCrossentropy()
        adv_loss = bce(tf.ones_like(fake_output) * (1.0 - self.config.label_smoothing), fake_output)
        
        fm_loss = self.feature_matching(real_features, fake_features)

        return fm_loss + self.config.adv_weight * adv_loss

    def discriminator_loss(self, real_output, fake_output):
        bce = tf.keras.losses.BinaryCrossentropy()
        real_loss = bce(tf.ones_like(real_output) * (1.0 - self.config.label_smoothing), real_output)
        fake_loss = bce(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.config.z_dim])

        with tf.GradientTape() as disc_tape:
            fake_data = self.generator(noise, training=True)
            real_output, real_features = self.discriminator(real_data, training=True, return_features=True)
            fake_output, fake_features = self.discriminator(fake_data, training=True, return_features=True)
            d_loss = self.discriminator_loss(real_output, fake_output)

        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            fake_output, fake_features = self.discriminator(fake_data, training=True, return_features=True)
            g_loss = self.generator_loss(fake_output, real_features, fake_features)

        gen_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        self.global_step += 1

        return {"gen_loss": g_loss, "disc_loss": d_loss}

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for batch in dataset:
                metrics = self.train_step(batch)
            print(f"Epoch {epoch+1}/{epochs} - Gen Loss: {metrics['gen_loss']:.4f}, Disc Loss: {metrics['disc_loss']:.4f}")



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
    bid_prices = np.array(lob_sample[:10])
    bid_sizes = np.array(lob_sample[10:20])
    ask_prices = np.array(lob_sample[20:30])
    ask_sizes = np.array(lob_sample[30:40])

    # Remove log/exp transformations and show raw price & quantity
    plt.figure(figsize=(8, 5))

    plt.scatter(bid_prices, bid_sizes, color='blue', label="Bids", alpha=0.6)
    plt.scatter(ask_prices, ask_sizes, color='red', label="Asks", alpha=0.6)

    for i, (p, q) in enumerate(zip(bid_prices, bid_sizes)):
        plt.annotate(f"b{i}", (p, q), fontsize=10, color='blue', ha="right")

    for i, (p, q) in enumerate(zip(ask_prices, ask_sizes)):
        plt.annotate(f"a{i}", (p, q), fontsize=10, color='red', ha="left")

    min_price = min(np.min(bid_prices), np.min(ask_prices)) * 0.99
    max_price = max(np.max(bid_prices), np.max(ask_prices)) * 1.01

    min_qty = min(np.min(bid_sizes), np.min(ask_sizes)) * 0.99
    max_qty = max(np.max(bid_sizes), np.max(ask_sizes)) * 1.01

    plt.xlim(min_price, max_price)
    plt.ylim(min_qty, max_qty)

    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title("LOB Scatter Plot (Original Scale)")
    plt.legend()
    plt.grid(True)
    # plt.show()



# ==========================================================================================
# train and generate
# ==========================================================================================

gan = SimpleGAN(config)
gan.train(lob_dataset, config.epochs)

noise = tf.random.normal([100, config.z_dim])
generated_samples = gan.generator(noise, training=False)  # Ignore the penalties
generated_samples = generated_samples.numpy()
# Convert back to original scale
generated_samples = scaler.inverse_transform(generated_samples.reshape(100, -1))

# ==========================================================================================
# reasonability check
# ==========================================================================================

faulty_rates = np.array([compute_faulty_rate(tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)) for sample in generated_samples])

# Create the directory if it doesn't exist
save_dir = "/Users/xuyunpeng/Documents/NUS/DSA5204/proj/GAN_LOB_Project/plots3"
os.makedirs(save_dir, exist_ok=True)

# 1. Save histogram of faulty rates
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(8, 5))
plt.hist(faulty_rates, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Faulty Rate")
plt.ylabel("Frequency")
plt.title("Histogram of Faulty Rates in Generated LOB Data")
plt.grid(True)

hist_filename = f"hist_faulty_rates_{timestamp}.png"
plt.savefig(os.path.join(save_dir, hist_filename))

# 2. Save the first generated LOB sample to CSV
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
print("First sample saved to 'single_generated_lob.csv'")

# 3. Plot and save scatter plots of order books
random_indices = random.sample(range(100), 5)

for i in random_indices:
    # Generate scatter plot
    plot_order_book_scatter(generated_samples[i])  
    
    # Get timestamp for unique naming
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct filename
    scatter_filename = f"order_book_scatter_{i}_{current_timestamp}.png"
    
    # Save figure
    plt.savefig(os.path.join(save_dir, scatter_filename))

