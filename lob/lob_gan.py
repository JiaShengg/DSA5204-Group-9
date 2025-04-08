import datetime
import time
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses


N_LOB_LEVELS = 5

PRICE_FEATURES = (
    [f'b{i}p' for i in reversed(range(N_LOB_LEVELS))]
    + [f'a{i}p' for i in range(N_LOB_LEVELS)]
)
QUANTITY_FEATURES = (
    [f'b{i}q' for i in reversed(range(N_LOB_LEVELS))]
    + [f'a{i}q' for i in range(N_LOB_LEVELS)]
)


@dataclass
class Configs:
    fm_multiplier: float = 0.0
    historical_averaging_weight: float = 0.0001
    use_minibatch_discrimination: bool = False
    label_smoothing: float = 0.0

    sample_size: int = 20_000
    n_batches: int = 16
    epochs: int = 64


N_FIXED_NOISE = 4
Z_DIM = 128


def create_dataset(raw_data, sample_size, batch_size,
                   scale=True, random_state=0):

    # scale across all columns to ensure relative ordering
    def _scale(dataf):
        max_val = dataf.max(axis=None)
        min_val = dataf.min(axis=None)
        scaled = dataf.copy(deep=True)
        scaled -= (max_val + min_val) / 2
        scaled /= (max_val - min_val) / 2
        return scaled

    dataf = (
        raw_data[PRICE_FEATURES + QUANTITY_FEATURES]
        .sample(n=sample_size, replace=False, random_state=random_state)
        .copy()
    )

    if scale:
        dataf[PRICE_FEATURES] = _scale(dataf[PRICE_FEATURES])

    return (
        tf.data.Dataset
        .from_tensor_slices(dataf.values.astype(np.float32))
        .batch(batch_size)
        # .shuffle(batch_size, seed=random_state, reshuffle_each_iteration=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def plot_lob_snapshot(lob_vector, ax) -> None:
    assert lob_vector.shape == (N_LOB_LEVELS * 4, ), lob_vector.shape

    prices = lob_vector[:2*N_LOB_LEVELS]
    quantities = lob_vector[2*N_LOB_LEVELS:]

    bidp = prices[:N_LOB_LEVELS][::-1]
    bidq = quantities[:N_LOB_LEVELS][::-1]
    askp = prices[N_LOB_LEVELS:]
    askq = quantities[N_LOB_LEVELS:]

    ax.plot(bidp, bidq, c='g', marker='x', alpha=0.4)
    ax.plot(askp, askq, c='r', marker='x', alpha=0.4)

    fontsize = 8
    for i, (p, q) in enumerate(zip(bidp, bidq)):
        ax.annotate(f'b{i}', (p, q),
                    fontsize=fontsize, color='green', ha='right')
    for i, (p, q) in enumerate(zip(askp, askq)):
        ax.annotate(f'a{i}', (p, q),
                    fontsize=fontsize, color='red', ha='left')

    ax.vlines(prices, 0, quantities, color='k', alpha=0.2)
    ax.axhline(0, color='k', alpha=0.2)
    ax.set_yscale('symlog', linthresh=0.001)
    ax.tick_params(axis='x', labelrotation=30)


def plot_lobs(all_lobs):
    for epoch, lobs in enumerate(all_lobs):
        if epoch % 10 != 0:
            continue
        f, axes = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
        f.suptitle(f'epoch {epoch}')
        (ax0, ax1), (ax2, ax3) = axes
        plot_lob_snapshot(lobs[0], ax0)
        plot_lob_snapshot(lobs[1], ax1)
        plot_lob_snapshot(lobs[2], ax2)
        plot_lob_snapshot(lobs[3], ax3)
        plt.tight_layout()
        plt.show()


def plot_training_history(metrics):
    print('plot_training_history')
    metrics_df = pd.DataFrame(metrics)
    display(metrics_df)  # type: ignore
    _, axes = plt.subplots(nrows=5, figsize=(6, 14), sharex=True)

    def plot_cols(ax, cols, log_scale=False):
        metrics_df[cols].plot(ax=ax, drawstyle='steps-post')
        if log_scale:
            ax.set_yscale('log')

    plot_cols(axes[0], ['gen_loss', 'adv_loss', 'fm_loss'], True)
    plot_cols(axes[1], ['disc_loss'], True)
    plot_cols(axes[2], ['real_prob', 'fake_prob'])
    plot_cols(axes[3], ['neg_qty_sum', 'neg_diff_sum'], True)
    plot_cols(axes[4], ['neg_qty_count', 'neg_diff_count'], True)
    plt.tight_layout()
    plt.show()


def calculate_sample_stats(batch) -> dict:
    quantities = batch[:, 2*N_LOB_LEVELS:]
    prices = batch[:, :2*N_LOB_LEVELS]
    price_diffs = prices[:, 1:] - prices[:, :-1]
    price_diffs = (
        tf.expand_dims(prices, axis=1) - tf.expand_dims(prices, axis=2)
    )
    price_diffs = tf.linalg.band_part(price_diffs, 0, -1)

    neg_qty_sum = tf.reduce_mean(
        tf.reduce_sum(tf.nn.relu(-quantities), axis=1)
    )
    neg_qty_count = tf.reduce_mean(
        tf.reduce_sum(tf.cast(tf.less(quantities, 0), tf.float32), axis=1)
    )

    neg_diff_sum = tf.reduce_mean(
        tf.reduce_sum(tf.nn.relu(-price_diffs), axis=(1, 2))
    )
    neg_diff_count = tf.reduce_mean(
        tf.reduce_sum(
            tf.cast(tf.less(price_diffs, 0), tf.float32), axis=(1, 2)
        )
    )
    return dict(
        neg_qty_sum=neg_qty_sum,
        neg_qty_count=neg_qty_count,
        neg_diff_sum=neg_diff_sum,
        neg_diff_count=neg_diff_count,
    )


def calculate_feature_matching_loss(real_lobs, fake_lobs):

    def calc_neg_px_diff_sums(lobs):
        prices = lobs[:, :2*N_LOB_LEVELS]
        price_diffs = prices[:, 1:] - prices[:, :-1]
        price_diffs = (
            tf.expand_dims(prices, axis=1) - tf.expand_dims(prices, axis=2)
        )
        price_diffs = tf.linalg.band_part(price_diffs, 0, -1)
        return tf.reduce_sum(tf.nn.relu(-price_diffs))

    def calc_neg_qty_sums(lobs):
        quantities = lobs[:, 2*N_LOB_LEVELS:]
        return tf.reduce_sum(tf.nn.relu(-quantities))

    def calc_px_stats(lobs):
        prices = lobs[:, :2*N_LOB_LEVELS]
        return (
            tf.reduce_mean(tf.math.reduce_variance(
                prices[N_LOB_LEVELS:], axis=1)),
            tf.reduce_mean(tf.math.reduce_variance(
                prices[:N_LOB_LEVELS], axis=1)),
            tf.reduce_max(prices) - tf.reduce_min(prices),
        )

    def calc_qty_stats(lobs):
        qtys = lobs[:, 2*N_LOB_LEVELS:]
        return (
            tf.reduce_mean(tf.math.reduce_variance(
                qtys[N_LOB_LEVELS:], axis=1)),
            tf.reduce_mean(tf.math.reduce_variance(
                qtys[:N_LOB_LEVELS], axis=1)),
            tf.reduce_max(qtys) - tf.reduce_min(qtys),
        )

    def calc_stats(lobs):
        return tf.stack([
            calc_neg_px_diff_sums(lobs),
            calc_neg_qty_sums(lobs),
            *calc_px_stats(lobs),
            *calc_qty_stats(lobs),
        ])

    return tf.norm(calc_stats(real_lobs) - calc_stats(fake_lobs))


class Generator(models.Model):
    def __init__(self, config):
        super().__init__()
        activation = 'softplus'
        self.px_net = models.Sequential([
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(N_LOB_LEVELS * 2, use_bias=False),
        ])
        self.qty_net = models.Sequential([
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(N_LOB_LEVELS * 2, use_bias=False),
        ])

    def call(self, inputs):
        x_price = inputs[:, :Z_DIM // 2]
        x_qty = inputs[:, Z_DIM // 2:]
        x_price = self.px_net(x_price)
        x_qty = self.qty_net(x_qty)
        return tf.concat([x_price, x_qty], axis=1)


class Discriminator(models.Model):
    def __init__(self, config):
        super().__init__()
        activation = 'softplus'
        self.px_net = models.Sequential([
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1),
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1),
            layers.Dense(1),
        ])
        self.qty_net = models.Sequential([
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1),
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1),
            layers.Dense(1),
        ])

    def call(self, inputs):
        x_price = inputs[:, :2 * N_LOB_LEVELS]
        x_qty = inputs[:, 2 * N_LOB_LEVELS:]
        x_price = self.px_net(x_price)
        x_qty = self.qty_net(x_qty)
        return tf.squeeze(x_price + x_qty)


class ImprovedGAN:
    def __init__(self, raw_data, config):
        self.config = config
        self.dataset = create_dataset(
            raw_data=raw_data,
            sample_size=min(len(raw_data), config.sample_size),
            batch_size=config.sample_size // config.n_batches,
        )
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.gen_optimizer = optimizers.Adam(learning_rate=0.01)
        self.disc_optimizer = optimizers.Adam(learning_rate=0.01)
        self.fixed_noise = tf.random.normal(
            [N_FIXED_NOISE, Z_DIM],
        )
        self.generated_lobs = []
        self.metrics = []

    @tf.function
    def train_step(self, real_lobs, epoch) -> dict:
        n_samples = tf.shape(real_lobs)[0]
        noise = tf.random.normal(
            shape=[n_samples, Z_DIM],
        )
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_lobs = self.generator(noise)
            real_output = self.discriminator(real_lobs)
            fake_output = self.discriminator(fake_lobs)
            real_labels = tf.ones_like(real_output)
            fake_labels = tf.zeros_like(fake_output)

            real_loss = tf.reduce_mean(losses.binary_crossentropy(
                real_labels, real_output, from_logits=True))
            fake_loss = tf.reduce_mean(losses.binary_crossentropy(
                fake_labels, fake_output, from_logits=True))
            disc_loss = real_loss + fake_loss

            adv_loss = tf.reduce_mean(losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output, from_logits=True))
            fm_loss = tf.math.log(1 + epoch) * self.config.fm_multiplier * \
                calculate_feature_matching_loss(real_lobs, fake_lobs)
            gen_loss = adv_loss + fm_loss

            fake_sample_stats = calculate_sample_stats(fake_lobs)

        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))

        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        real_output = self.discriminator(real_lobs)
        fake_output = self.discriminator(self.generator(noise))
        real_prob = tf.reduce_mean(tf.sigmoid(real_output))
        fake_prob = tf.reduce_mean(tf.sigmoid(fake_output))
        return dict(
            disc_loss=disc_loss,
            gen_loss=gen_loss,
            adv_loss=adv_loss,
            fm_loss=fm_loss,
            real_prob=real_prob,
            fake_prob=fake_prob,
            **fake_sample_stats,
        )

    def train(self):
        g1 = self.generator.predict(self.fixed_noise, verbose=False)
        g2 = self.generator.predict(self.fixed_noise, verbose=False)
        assert tf.reduce_all(g1 == g2)

        start_time = time.time()

        for epoch in tqdm(range(self.config.epochs)):
            self.generated_lobs.append(
                self.generator.predict(self.fixed_noise, verbose=False))

            epoch_metrics_list = []
            for batch in self.dataset:
                batch = tf.identity(batch)  # move to gpu
                metrics = self.train_step(
                    batch, tf.constant(epoch, dtype=tf.float32))
                metrics = {k: float(v) for k, v in metrics.items()}
                # insert historical averaging here
                epoch_metrics_list.append(metrics)

            epoch_metrics = pd.DataFrame(epoch_metrics_list).mean()
            self.metrics.append(epoch_metrics)

        self.generated_lobs.append(
            self.generator.predict(self.fixed_noise, verbose=False))

        total_time = time.time() - start_time
        print(f'Training completed in {total_time/60:.2f} minutes')
        return self.generated_lobs, self.metrics


print(f'imported at {datetime.datetime.now()}')
