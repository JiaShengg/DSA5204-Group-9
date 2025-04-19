from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from pathlib import Path


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
    fm_weight_h: float = 0.0  # handmade features
    fm_weight_e: float = 0.0  # extracted features

    historical_averaging_weight: float = 0.0001
    use_minibatch_discrimination: bool = False
    label_smoothing: float = 0.0

    gen_lr: float = 0.001
    disc_lr: float = 0.0001

    z_dim = 64
    sample_size: int = 20_000
    n_batches: int = 16
    epochs: int = 64
    seed: int = 0


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
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
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


def plot_training_history(metrics: pd.DataFrame):
    print('plot_training_history')
    display(metrics)  # type: ignore
    _, axes = plt.subplots(nrows=5, figsize=(6, 14), sharex=True)

    def plot_cols(ax, cols, log_scale=False):
        metrics[cols].plot(ax=ax, drawstyle='steps-post')
        if log_scale:
            ax.set_yscale('log')

    plot_cols(axes[0],
              ['gen_loss', 'adv_loss', 'fm_loss_h', 'fm_loss_e'], True)
    plot_cols(axes[1], ['disc_loss', 'real_loss', 'fake_loss'], True)
    plot_cols(axes[2], ['real_prob', 'fake_prob'])
    plot_cols(axes[3], ['neg_qty_sum', 'neg_diff_sum'], True)
    plot_cols(axes[4], ['neg_qty_count', 'neg_diff_count'])
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
        neg_qty_sum=float(neg_qty_sum),
        neg_qty_count=float(neg_qty_count),
        neg_diff_sum=float(neg_diff_sum),
        neg_diff_count=float(neg_diff_count),
    )


class Generator(models.Model):
    def __init__(self, config):
        super().__init__()
        activation = 'relu'
        self.config = config
        self.px_net = models.Sequential([
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(N_LOB_LEVELS * 2, use_bias=False),
        ])
        self.qty_net = models.Sequential([
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(64, activation=activation, use_bias=False),
            layers.Dense(N_LOB_LEVELS * 2, use_bias=True),
        ])

    def call(self, inputs):
        x_price = inputs[:, :self.config.z_dim // 2]
        x_qty = inputs[:, self.config.z_dim // 2:]
        x_price = self.px_net(x_price)
        x_qty = self.qty_net(x_qty)
        return tf.concat([x_price, x_qty], axis=1)


# column vector of summary statistics for the batch
def _calculate_handmade_features(prices, qtys):
    # price_diffs = prices[:, 1:] - prices[:, :-1]
    price_diffs = (
        tf.expand_dims(prices, axis=1) - tf.expand_dims(prices, axis=2)
    )
    price_diffs = tf.linalg.band_part(price_diffs, 0, -1)

    return tf.stack([
        tf.reduce_sum(tf.nn.relu(-price_diffs)),
        tf.reduce_sum(tf.nn.relu(-qtys)),

        tf.reduce_mean(tf.math.reduce_std(prices, axis=1)),
        tf.reduce_min(price_diffs),
        tf.reduce_max(price_diffs),

        tf.reduce_mean(tf.math.reduce_mean(qtys, axis=1)),
        tf.reduce_mean(tf.math.reduce_std(qtys, axis=1)),
        tf.reduce_min(qtys),
        tf.reduce_max(qtys),
    ], axis=0)


class MinibatchDiscrimination(layers.Layer):
    """Computes pairwise difference features to prevent mode collapse.

    Simplified to only directly compute pairwise distances without learning any
    additional parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x1 = tf.expand_dims(inputs, axis=1)  # shape (n, 1, p)
        x2 = tf.expand_dims(inputs, axis=0)  # shape (1, n, p)
        l1_distances = tf.reduce_sum(tf.abs(x1 - x2), axis=-1)  # shape (n, n)
        return tf.concat([inputs, l1_distances], axis=-1)


class Discriminator(models.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        activation = 'relu'

        if config.use_minibatch_discrimination:
            FeatureLayer = MinibatchDiscrimination
        else:
            FeatureLayer = layers.Identity

        self.px_layers = [
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1, seed=config.seed),
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1, seed=config.seed),
            FeatureLayer(name='p_features'),
            layers.Dense(1),
        ]
        self.qty_layers = [
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1, seed=config.seed),
            layers.Dense(64, activation=activation),
            layers.Dropout(0.1, seed=config.seed),
            FeatureLayer(name='q_features'),
            layers.Dense(1),
        ]
        assert len(self.px_layers) == len(self.qty_layers), \
            (len(self.px_layers), len(self.qty_layers))

    def call(self, inputs):

        x_price = inputs[:, :2 * N_LOB_LEVELS]
        x_qty = inputs[:, 2 * N_LOB_LEVELS:]

        handmade_features = tf.constant(0, dtype=tf.float32, shape=(1, ))
        extracted_features = tf.constant(0, dtype=tf.float32, shape=(1, ))

        if self.config.fm_weight_h:
            handmade_features = _calculate_handmade_features(x_price, x_qty)

        for p_layer, q_layer in zip(self.px_layers, self.qty_layers):
            x_price = p_layer(x_price)
            x_qty = q_layer(x_qty)
            if not self.config.fm_weight_e:
                continue
            if 'features' not in p_layer.name or 'features' not in q_layer.name:
                continue
            assert 'features' in p_layer.name and 'features' in q_layer.name, \
                (p_layer.name, q_layer.name)
            extracted_features = tf.math.reduce_mean(
                tf.concat([x_price, x_qty], axis=1), axis=0)

        prediction = tf.squeeze(x_price + x_qty)
        return prediction, handmade_features, extracted_features


@dataclass(eq=False, frozen=True, slots=True)
class TrainingOutputs:
    metrics: pd.DataFrame
    lobs: list[np.ndarray]

    def dump(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        self.metrics.to_parquet(path / f'metrics.parq')
        for i, lob in enumerate(self.lobs):
            np.save(path / f'lob-{i:03}.npy', lob)

    @staticmethod
    def load(path: Path) -> TrainingOutputs:
        return TrainingOutputs(
            metrics=pd.read_parquet(path / f'metrics.parq'),
            lobs=[np.load(p) for p in sorted(path.glob('*.npy'))],
        )

    def __eq__(self, rhs) -> bool:
        if (self.metrics != rhs.metrics).any(axis=None):
            return False
        for l, r in zip(self.lobs, rhs.lobs):
            if (l != r).any():
                return False
        return True


class ImprovedGAN:
    def __init__(self, raw_data, config):
        tf.keras.utils.set_random_seed(config.seed)

        batch_size = config.sample_size // config.n_batches
        print(f'{batch_size=}')
        self.config = config
        self.dataset = create_dataset(
            raw_data=raw_data,
            sample_size=min(len(raw_data), config.sample_size),
            batch_size=batch_size,
        )
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.gen_optimizer = optimizers.AdamW(learning_rate=config.gen_lr)
        self.disc_optimizer = optimizers.AdamW(learning_rate=config.disc_lr)
        self.fixed_noise = tf.random.normal(
            [batch_size, config.z_dim],
            seed=config.seed,
        )
        self.step = 0

    @tf.function
    def train_step(self, real_lobs) -> dict:
        n_samples = tf.shape(real_lobs)[0]

        ones = tf.ones(n_samples)
        zeros = tf.ones(n_samples)

        with tf.GradientTape() as gen_tape:
            adv_loss = 0
            fm_loss_h = 0
            fm_loss_e = 0

            for _ in range(1):
                noise = tf.random.normal(
                    shape=[n_samples, self.config.z_dim],
                    seed=self.config.seed + self.step,
                )
                self.step += 1
                fake_lobs = self.generator(noise)

                real_output, real_h_feats, real_e_feats = self.discriminator(
                    real_lobs)
                fake_output, fake_h_feats, fake_e_feats = self.discriminator(
                    fake_lobs)

                adv_loss += 10 * tf.reduce_mean(losses.binary_crossentropy(
                    ones, fake_output, from_logits=True))
                fm_loss_h += self.config.fm_weight_h * \
                    tf.norm(real_h_feats - fake_h_feats)
                fm_loss_e += self.config.fm_weight_h * \
                    tf.norm(real_e_feats - fake_e_feats)

            gen_loss = adv_loss + fm_loss_h + fm_loss_e

        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        with tf.GradientTape() as disc_tape:
            noise = tf.random.normal(
                shape=[n_samples, self.config.z_dim],
                seed=self.config.seed + self.step,
            )
            self.step += 1
            fake_lobs = self.generator(noise)
            real_output, _, _ = self.discriminator(real_lobs)
            fake_output, _, _ = self.discriminator(fake_lobs)

            real_loss = tf.reduce_mean(losses.binary_crossentropy(
                ones, real_output, from_logits=True))
            fake_loss = tf.reduce_mean(losses.binary_crossentropy(
                zeros, fake_output, from_logits=True))
            disc_loss = real_loss + fake_loss

        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))

        return dict(
            disc_loss=disc_loss,
            real_loss=real_loss,
            fake_loss=fake_loss,
            gen_loss=gen_loss,
            adv_loss=adv_loss,
            fm_loss_h=fm_loss_h,
            fm_loss_e=fm_loss_e,
            real_prob=tf.reduce_mean(tf.sigmoid(real_output)),
            fake_prob=tf.reduce_mean(tf.sigmoid(fake_output)),
        )

    def train(self) -> TrainingOutputs:
        start_time = time.time()
        lobs_list = []
        metrics_list = []

        lobs_list.append(self.generator.predict(
            self.fixed_noise, verbose=False))

        for epoch in tqdm(range(self.config.epochs)):
            epoch_metrics_list = []
            for batch in self.dataset:
                metrics = self.train_step(batch)
                metrics = {k: float(v) for k, v in metrics.items()}
                # insert historical averaging here
                epoch_metrics_list.append(metrics)

            fake_batch = self.generator.predict(
                self.fixed_noise, verbose=False)
            lobs_list.append(fake_batch)
            sample_stats = calculate_sample_stats(fake_batch)
            epoch_metrics = dict(
                pd.DataFrame(epoch_metrics_list).mean(),
                **sample_stats,
            )
            metrics_list.append(epoch_metrics)

        total_time = time.time() - start_time
        print(f'Training completed in {total_time/60:.2f} minutes')
        all_metrics = pd.DataFrame(metrics_list)
        return TrainingOutputs(metrics=all_metrics, lobs=lobs_list)


print(f'imported at {datetime.datetime.now()}')
