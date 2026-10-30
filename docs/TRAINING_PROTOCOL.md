# Training protocol: Flowers102

## Recorded experiment

Three completed runs compare Linear probing, LoRA and SSF on 800 training and 200 validation images. Each uses seed 42, supervised ImageNet-21K ViT-B/16 and 100 epochs. Epoch-100 validation accuracy is the primary metric. Best-validation checkpoints are diagnostic.

The recorded environment uses Python 3.10.4, PyTorch 1.13.0+cu117, torchvision 0.14.0+cu117 and timm 0.9.12. Per-run provenance records the original source hashes, versions, split hashes, initialization hashes and runtime policy. These records are retained without rewriting them to match later source cleanup.

## Data and model setup

The CPU analysis environment is separate from the GPU training environment. Training requires Linux, an NVIDIA GPU, Python 3.10 and uv on PATH. PEFT_TRAINING_PYTHON can select the Python executable used to create the environment. PEFT_UV can select the uv executable.

The source and model weights are pinned through the preparation scripts:

```bash
bash scripts/get_upstream.sh
bash scripts/setup_training.sh
.venv/bin/python scripts/prepare_vtab.py
```

The setup script installs the recorded GPU dependencies and prepares ImageNet-21K weights. Dataset preparation downloads the pinned VTAB mirror archive and checks its checksum and split structure. These operations are separate from notebook Run All.

Preprocessing uses resize bicubic 224 x 224 and ToTensor, with normalized=False and no image augmentation. Model drop path is 0.1. Native LoRA input dropout is 0.1 with merge_factor=1.

## Optimizer and batching

All methods use AdamW, LR 0.001, weight decay 0.0001, cosine decay, warmup 10 epochs from LR 1e-6, and minimum LR 1e-5. LoRA rank is 8. These values are fixed baseline choices within the paper grid, not tuned optima.

Linear uses batch 64 directly. LoRA and SSF use effective batch 64 with microbatch 8. Cross-entropy microbatch losses are weighted by microbatch size divided by effective-batch size. There is one optimizer update per effective batch.

The loader uses shuffle and drop_last=True. Each epoch uses 768 images and 12 optimizer updates. The sampler is independently seeded per epoch. Gradient accumulation is checked on deterministic CPU models, while stochastic ViT trajectories are not asserted to match direct batching.

## Runner entrypoint

The source supports independent runs without an existing result folder. An optional reference result checks initial classifier and frozen-backbone hashes:

```bash
.venv/bin/python scripts/train_validation.py --method linear --config configs/flowers102_stage3.json --output runs/flowers102/linear
.venv/bin/python scripts/train_validation.py --method lora --config configs/flowers102_stage3_microbatch8.json --output runs/flowers102/lora --reference-result runs/flowers102/linear/result.json
.venv/bin/python scripts/train_validation.py --method ssf --config configs/flowers102_ssf.json --output runs/flowers102/ssf --reference-result runs/flowers102/linear/result.json
```

The SSF recipe extracts the same SSF settings from the original combined queue configuration. Historical result/config JSON files retain their original contents.

A missing explicit reference fails before resource checks. A mismatched initialization fails before moving the model to GPU. Omitting the reference only omits the cross-run comparison. Frozen-backbone checks and within-run checkpoint checks remain enabled.

## Resource policy and resume

Training uses FP32, zero loader workers, no pinned CPU memory and one CPU thread. Recorded LoRA and SSF runs use allocator trimming under the shared node's 16 GiB host-RAM limit. Resource guards remain in the runner. Startup thresholds are admission rules, not measurements of a model's intrinsic requirement.

Checkpoint state contains trainable weights, optimizer, scheduler, RNG state and history. Resume verifies the saved configuration and initialization hashes. Final exports compare logits before and after checkpoint reload. Frozen backbone hashes are checked around training.

Total epoch duration sums the measured training-plus-validation intervals. It excludes unmeasured setup, queue delays and some checkpoint overhead. Mean epoch duration is reported separately. Original memory and duration observations are not a controlled efficiency ranking.

The runner's optional-reference and path-portability fixes were made after the recorded experiments. No training was repeated for the checkpoint report. Independent test evaluation and trainval retraining remain pending.
