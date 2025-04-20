from pathlib import Path

import numpy as np
import pandas as pd
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
            use_minibatch_discrimination=mbd,
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
