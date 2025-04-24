# Limit Order Book Extension for GAN

This directory contains the code for extending GAN improvements to generate Limit Order Book snapshots.

`lob_gan.py` explores the following GAN improvement techniques:
* Feature Matching
* Minibatch Discrimination
* Label Smoothing

`param_search.py` contains code for hyperparameter and logic for
* saving every model's configurations, traning results, generated LOB samples to ensure reproducibility
* caching results to avoid retraining models for computational efficiency

Other files:
* `BTCUSDT-lob.parq`: bitcoin LOB snapshots
* `ETHUSDT-lob.parq`: ethereum LOB snapshots
* `experiment.ipynb`: runs parameter search and visualises results
* `lob_gan.ipynb`: trains model and visualises results
* `raw_lob_snapshot_row0.png`: sample visualisation of LOB snapshot