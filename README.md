# Visual PEFT on Flowers102

**Kelompok 1: Alif Negifta Wibawaputra, Charleen Samaeri Gulo**

Reproduction progress for [Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning in Visual Recognition](https://arxiv.org/abs/2409.16434), CVPR 2025.

The [notebook](notebooks/01_flowers102_checkpoint1.ipynb) reports the dataset audit, ViT and adaptation methods, training protocol, results, error analysis and proposed extensions.

## Progress

- Linear probing, LoRA and SSF completed 100 epochs on the same 800-image training and 200-image validation split.
- Stored logits, histories, initialization hashes and checkpoint-reload predictions were checked.
- CPU analysis includes macro recall, NLL, Brier, reliability plots, paired bootstrap and two ensemble aggregation rules.
- Hyperparameter search, repeated training seeds, trainval retraining and final test evaluation remain pending.
- ConvPass stopped at epoch 92 and is excluded from the completed-method comparison.

## Configuration

The runs use supervised ImageNet-21K ViT-B/16, seed 42, AdamW, learning rate 0.001, weight decay 0.0001, effective batch 64, cosine decay, 10 warmup epochs and drop path 0.1. LoRA uses rank 8. LoRA and SSF use microbatch 8. All runs use FP32.

Learning rate, weight decay and rank are baseline choices within the official grid. They have not been tuned in this experiment. Native LoRA updates Q, K and V through a shared low-rank projection. See [configuration notes](docs/CONFIGURATION_NOTES.md) and the [training protocol](docs/TRAINING_PROTOCOL.md).

## Validation results

All three methods correctly classify **198 of 200 images: 99% top-1 accuracy at epoch 100**.

- Linear: NLL 0.1162, multiclass Brier 0.03572, 78,438 trainable parameters.
- LoRA: NLL 0.04099, Brier 0.01373, 373,350 trainable parameters.
- SSF: NLL 0.03558, Brier 0.01345, 284,262 trainable parameters.

Parameter counts include the classifier. Brier sums over 102 classes before averaging over images. Macro recall averages only over the 90 classes observed in validation.

Uniform probability averaging and logit averaging both give 200/200 correct for Linear-LoRA and 199/200 for the other three combinations. Their NLL values differ. These are exploratory results on reused validation data.

SSF has the lowest single-model NLL point estimate. The paired-bootstrap interval for SSF minus LoRA includes zero. The experiment does not establish an overall winner.

## Run the CPU notebook

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --user --name peft-checkpoint1-cpu --display-name "PEFT Checkpoint 1 (CPU)"
```

Open the notebook, select **PEFT Checkpoint 1 (CPU)**, restart the kernel and Run All. Start from the repository root or the notebooks folder. Analysis reads local snapshots and does not start training or download data.

Figures and CSV tables are regenerated under assets/checkpoint1. The asset index records sources, producing sections, interpretation limits and hashes. Training code is separate and its GPU environment is documented in the training protocol.

CPU checks:

```bash
python -m unittest discover -s tests -p 'test_checkpoint*.py' -v
```

## Interpretation limits

This is a fixed-configuration, single-seed comparison on one VTAB-1K task. Twelve classes have no validation samples. Exact mirror split membership against the authors' archive is unverified.

Test metadata and sample files were checked during integrity preparation. Test data were not used for training, configuration selection or performance evaluation. Validation scores are not directly comparable with the paper's tuned and retrained test results. Runtime and memory logs come from shared-node conditions.

Proposed extensions cover selective fusion, class-preserving weighting under controlled label noise, and resource-aware recipe selection. Their evaluation plans and literature context are in [research directions](docs/RESEARCH_DIRECTIONS.md). None has been evaluated yet.

## Sources

- [Paper](https://arxiv.org/abs/2409.16434) and [official code](https://github.com/OSU-MLB/ViT_PEFT_Vision), commit cea7e21b2ddecef8172f5d177da5a29d531c4279.
- [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [VTAB-1K mirror](https://huggingface.co/datasets/XiN0919/VTAB-1k), revision a1c257fbbeebd3260c0a201fe6a6c9916b56eee7.
- [Image attribution](assets/checkpoint1/ATTRIBUTION.md), [reference metadata](docs/references.json) and [verification record](docs/CHECKPOINT1_VERIFICATION.json).
