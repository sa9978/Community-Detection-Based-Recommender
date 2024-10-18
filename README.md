# Community-Detection-Based Recommender

A research implementation for studying whether graph-community assignments can improve node-to-node recommendation. The project detects communities in an attributed graph, derives community-aware features, and evaluates several recommendation architectures.

## Methods

| Stage | Implementations |
| --- | --- |
| Community detection | NOCD, SEComm, VGAER |
| Feature processing | Community-wise CNN aggregation followed by an MLP |
| Recommendation | PinSAGE-style model, lightweight GraphSAGE, NGCF, and community-aware NGCF |
| Evaluation | Accuracy, NMI, macro F1 for communities; precision, recall, MAP, and NDCG for recommendations |

## Repository layout

```text
CD_models/             Community-detection implementations and SEComm configs
data_loader.py         PyTorch Geometric dataset loader
aggregator.py          Community-aware feature construction
recommendation*.py     GraphSAGE/PinSAGE recommendation experiments
ngcf.py, cd_ngcf.py    NGCF baselines and community-feature variant
main.py                Community-detection command-line entry point
RESULTS.md             Recorded SEComm community-detection results
```

Generated datasets, checkpoints, TensorBoard logs, labels, and feature arrays are deliberately ignored by Git. They can be regenerated locally and historical copies remain in the local `archive-original-history` branch after the repository cleanup.

## Setup

Use Python 3.11 and install a PyTorch build appropriate for your hardware first. Then install the remaining dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

The PyTorch Geometric installation may require a wheel matching your PyTorch/CUDA version; follow its official installation guidance when needed.

## Run community detection

```bash
python main.py --method secomm --dataset citeseer
```

Available methods are `nocd`, `secomm`, and `vgaer`. Available datasets are defined in `data_loader.py`: `cora`, `citeseer`, `pubmed`, `lastfmasia`, `amazon_photos`, `amazon_computers`, `coauthor_cs`, and `coauthor_physics`.

The first execution downloads a dataset through PyTorch Geometric. Community labels are written to `Communities/`, TensorBoard logs to `runs/`, and SEComm metrics to `Metrics/`.

## Reproduce the feature/recommendation pipeline

After a community-detection run, create community-aware features:

```bash
python aggregator.py
```

The recommendation scripts are experimental entry points with their dataset and model choices declared near the bottom of each file:

- `recommendation.py`: PinSAGE-style recommender
- `recommendation_lighter.py`: lightweight GraphSAGE recommender
- `ngcf.py`: graph-only NGCF baseline
- `cd_ngcf.py`: NGCF initialized with community-aware features

## Notes on reproducibility

SEComm has dataset-specific hyperparameters in `CD_models/SEComm/config.yaml`. Its training sets a configured Torch and Python random seed. Results in `RESULTS.md` are historic experiment outputs, not fresh benchmark claims.

## License

No license has been selected yet. Add one before distributing or accepting external contributions.
