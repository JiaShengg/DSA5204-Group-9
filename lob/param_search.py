from pathlib import Path

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from scipy.stats import loguniform

from lob_gan import Config, ImprovedGAN


def _generate_grid_impl(grid_type: str, size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed=seed)
    common = dict(
        gen_lr=loguniform.rvs(1e-3, 1e-2, size=size, random_state=rng),
        disc_lr=loguniform.rvs(1e-3, 1e-2, size=size, random_state=rng),
        z_dim=rng.choice([32, 64, 128, 256], size=size),
        n_batches=rng.choice([8, 16, 32], size=size),
    )
    mbd = rng.choice([True, False], size=size)
    fm_h = loguniform.rvs(1e-4, 1e1, size=size, random_state=rng)
    fm_e = loguniform.rvs(1e-4, 1e1, size=size, random_state=rng)
    ls = loguniform.rvs(1e-3, 1e-1, size=size, random_state=rng)

    if grid_type == 'baseline':
        return common
    if grid_type == 'fm_only':
        return dict(
            **common,
            fm_weight_h=fm_h,
            fm_weight_e=fm_e,
        )
    if grid_type == 'mbd_only':
        return dict(
            **common,
            use_minibatch_discrimination=True,
        )
    if grid_type == 'ls_only':
        return dict(
            **common,
            label_smoothing=ls,
        )
    if grid_type == 'all':
        return dict(
            **common,
            fm_weight_h=fm_h,
            fm_weight_e=fm_e,
            use_minibatch_discrimination=mbd,
            label_smoothing=ls,
        )
    assert False, f'unexpected grid_type: {grid_type}'


def generate_configs(grid_type: str, *, size: int, seed: int) -> list[Config]:
    grid = _generate_grid_impl(grid_type=grid_type, size=size, seed=seed)
    df = pd.DataFrame.from_dict(grid)
    return [Config(**d, seed=seed)  # type: ignore
            for d in df.to_dict(orient='records')]


def train_model(output_dir: Path, raw_data: pd.DataFrame, config: Config):
    key = config.get_key()

    model_dir = output_dir / key
    model_dir.mkdir(exist_ok=True)
    if (model_dir / 'metrics.parq').exists():
        print(f'results cached, skipping {key}')
        return

    with open(model_dir / 'config.json', 'w') as f:
        f.write(config.to_json(indent=2))
    gan = ImprovedGAN(raw_data, config)
    outputs = gan.train()
    outputs.dump(model_dir)
    print(f'finished trainining {key}')

def export_model_outputs(output_root: Path, plot_lob_snapshot):
    all_model_dirs = sorted([d for d in output_root.iterdir() if d.is_dir() and d.name != "image"])

    for model_dir in all_model_dirs:
        print(f"\nProcessing {model_dir.name}...")

        config_path = model_dir / "config.json"
        metrics_path = model_dir / "metrics.parq"

        # Skip if required files are missing
        if not config_path.exists() or not metrics_path.exists():
            print("Skipping: Missing config.json or metrics.parq")
            continue

        # === 1. Export config.json
        with open(config_path) as f:
            config = json.load(f)

        with open(model_dir / "config_pretty.json", "w") as f:
            json.dump(config, f, indent=2)

        with open(model_dir / "config.txt", "w") as f:
            for k, v in config.items():
                f.write(f"{k}: {v}\n")

        print("Exported config")

        # === 2. Export metrics.parq
        metrics = pd.read_parquet(metrics_path)
        metrics.to_csv(model_dir / "metrics.csv", index=False)
        metrics.to_excel(model_dir / "metrics.xlsx", index=False)

        print("Exported metrics")

        # === 3. Export lob-*.npy as images
        npy_files = sorted(model_dir.glob("lob-*.npy"))
        output_img_dir = model_dir / "images"
        output_img_dir.mkdir(exist_ok=True)

        for npy_file in npy_files:
            epoch = int(npy_file.stem.split("-")[1])
            data = np.load(npy_file)

            fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
            fig.suptitle(f"Epoch {epoch}")

            for i, ax in enumerate(axes.flat):
                if i < len(data):
                    plot_lob_snapshot(data[i], ax)
                else:
                    ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_img_dir / f"epoch-{epoch:03}.png")
            plt.close()

        print(f"Exported {len(npy_files)} LOB snapshots to images")

    print("\nAll model outputs processed.")


def plot_all_model_metrics(output_root: Path):
    """
    Reads all metrics.csv files in output_root, and generates:
    - Per-model plots for each metric in output_root/image/<metric>/
    - Combined real_prob + fake_prob charts in output_root/image/real_fake_prob/
    """

    image_output = output_root / "image"
    image_output.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "disc_loss", "real_loss", "fake_loss", "gen_loss", "adv_loss",
        "fm_loss_h", "fm_loss_e", "real_prob", "fake_prob",
        "neg_qty_sum", "neg_qty_count", "neg_diff_sum", "neg_diff_count"
    ]

    metric_data = {metric: {} for metric in metric_columns}

    # === Load metrics.csv from each model folder ===
    for model_dir in output_root.iterdir():
        if not model_dir.is_dir() or model_dir.name == "image" or not (model_dir / "metrics.csv").exists():
            continue

        model_name = model_dir.name
        metrics_df = pd.read_csv(model_dir / "metrics.csv")

        for metric in metric_columns:
            if metric in metrics_df.columns:
                df = pd.DataFrame({
                    "epoch": range(len(metrics_df)),
                    "value": metrics_df[metric]
                })
                metric_data[metric][model_name] = df

    # === Step 2: Plot each metric per model ===
    for metric, model_dict in metric_data.items():
        if not model_dict or metric in ["real_prob", "fake_prob"]:
            continue  # skip until next block

        metric_folder = image_output / metric
        metric_folder.mkdir(parents=True, exist_ok=True)

        all_epochs = []
        all_values = []
        for df in model_dict.values():
            all_epochs.extend(df["epoch"].values)
            all_values.extend(df["value"].values)

        x_min, x_max = min(all_epochs), max(all_epochs)
        y_min, y_max = min(all_values), max(all_values)

        for model_name, df in model_dict.items():
            plt.figure(figsize=(10, 5))
            plt.plot(df["epoch"], df["value"], label=model_name)
            plt.title(f"{metric} for {model_name}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            plt.grid(True)
            plt.tight_layout()

            save_path = metric_folder / f"{model_name}.png"
            plt.savefig(save_path)
            plt.close()

    # === Special case: combined real_prob + fake_prob ===
    real_fake_folder = image_output / "real_fake_prob"
    real_fake_folder.mkdir(parents=True, exist_ok=True)

    real_dict = metric_data.get("real_prob", {})
    fake_dict = metric_data.get("fake_prob", {})

    matched_models = [model for model in real_dict if model in fake_dict]

    if not matched_models:
        print("No matching models with both real_prob and fake_prob. Skipping real_fake_prob plots.")
        return

    all_epochs = []
    all_values = []
    for model in matched_models:
        all_epochs.extend(real_dict[model]["epoch"].values)
        all_values.extend(real_dict[model]["value"].values)
        all_values.extend(fake_dict[model]["value"].values)

    x_min, x_max = min(all_epochs), max(all_epochs)
    y_min, y_max = min(all_values), max(all_values)

    for model_name in matched_models:
        df_real = real_dict[model_name]
        df_fake = fake_dict[model_name]

        plt.figure(figsize=(10, 5))
        plt.plot(df_real["epoch"], df_real["value"], label="real_prob", color="green")
        plt.plot(df_fake["epoch"], df_fake["value"], label="fake_prob", color="red")
        plt.title(f"real vs fake prob for {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = real_fake_folder / f"{model_name}.png"
        plt.savefig(save_path)
        plt.close()

def compare_models(output_root: Path):
    """
    For each metric, generate a single plot comparing all models.
    - Regular metrics go into output_root/image/data_compare/<metric>.png
    - real_prob and fake_prob are combined into one plot: real_fake_prob_comparison.png
    """

    compare_folder = output_root / "image" / "data_compare"
    compare_folder.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "disc_loss", "real_loss", "fake_loss", "gen_loss", "adv_loss",
        "fm_loss_h", "fm_loss_e", "real_prob", "fake_prob",
        "neg_qty_sum", "neg_qty_count", "neg_diff_sum", "neg_diff_count"
    ]

    metric_data = {metric: {} for metric in metric_columns}

    # === Load metrics.csv from each model folder ===
    for model_dir in output_root.iterdir():
        if not model_dir.is_dir() or not (model_dir / "metrics.csv").exists():
            continue

        model_name = model_dir.name
        metrics_df = pd.read_csv(model_dir / "metrics.csv")

        for metric in metric_columns:
            if metric in metrics_df.columns:
                df = pd.DataFrame({
                    "epoch": range(len(metrics_df)),
                    "value": metrics_df[metric]
                })
                metric_data[metric][model_name] = df

    # === Plot one chart per metric with all models ===
    for metric, model_dict in metric_data.items():
        if not model_dict or metric in ["real_prob", "fake_prob"]:
            continue

        plt.figure(figsize=(12, 6))
        for model_name, df in model_dict.items():
            plt.plot(df["epoch"], df["value"], label=model_name)

        plt.title(f"{metric} comparison across models")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = compare_folder / f"{metric}.png"
        plt.savefig(save_path)
        plt.close()

    # === Special combined chart: real_prob and fake_prob ===
    real_dict = metric_data.get("real_prob", {})
    fake_dict = metric_data.get("fake_prob", {})

    plt.figure(figsize=(12, 6))
    for model_name in real_dict:
        if model_name in fake_dict:
            plt.plot(real_dict[model_name]["epoch"], real_dict[model_name]["value"],
                     label=f"{model_name} - real", linestyle='--')
            plt.plot(fake_dict[model_name]["epoch"], fake_dict[model_name]["value"],
                     label=f"{model_name} - fake", linestyle='-')

    plt.title("real vs fake prob comparison across models")
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = compare_folder / "real_fake_prob_comparison.png"
    plt.savefig(save_path)
    plt.close()

    print("Exported all comparison plots to 'data_compare'")



def main():
    GRID_TYPE = 'baseline'
    FILE_PATH = Path('lob/BTCUSDT-lob.parq')
    OUTPUT_DIR = Path('output')
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_data = pd.read_parquet(FILE_PATH, engine='pyarrow')
    configs = generate_configs(GRID_TYPE, size=30, seed=0)

    for config in configs:
        train_model(output_dir=OUTPUT_DIR, raw_data=raw_data, config=config)
        break


if __name__ == '__main__':
    main()
