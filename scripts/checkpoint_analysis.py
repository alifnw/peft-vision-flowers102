"""CPU analysis of immutable Checkpoint 1 snapshots, without training imports."""

from pathlib import Path
from itertools import combinations
from statistics import NormalDist
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.special import logsumexp


METHODS = ("Linear", "LoRA", "SSF")
COLORS = {"Linear": "#465D86", "LoRA": "#C47723", "SSF": "#387A68"}
DATASET_CREDIT = "Images: Oxford Flowers102, Nilsback & Zisserman. VTAB-1K mirror: XiN0919."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset_snapshot(root):
    folder = Path(root) / "assets/checkpoint1/input"
    audit = json.loads((folder / "dataset_audit.json").read_text())
    records = {name: pd.read_csv(folder / f"{name}_rows.csv")
               for name in ("train", "validation")}
    for name, frame in records.items():
        expected = audit["splits"][name]
        assert len(frame) == expected["rows"]
        assert frame["row"].tolist() == list(range(1, len(frame) + 1))
        actual = frame["label"].value_counts().sort_index().to_dict()
        assert actual == {int(k): v for k, v in expected["class_counts"].items()}
    for item in audit["images"]:
        image = folder / item["file"]
        assert sha256(image) == item["sha256"], f"Image hash mismatch: {image}"
        with Image.open(image) as photo:
            photo.verify()
        row = records[item["split"]].iloc[item["row"] - 1]
        assert int(row["label"]) == item["label"]
        assert row["original_relative_path"] == item["original_relative_path"]
    return {"folder": folder, "audit": audit, "records": records}


class AssetExporter:
    """Save figures and tables and record their evidence and interpretation limits."""

    def __init__(self, root):
        self.root = Path(root)
        self.folder = self.root / "assets/checkpoint1"
        for name in ("figures", "tables"):
            (self.folder / name).mkdir(parents=True, exist_ok=True)
        self.records = []
        plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "peft-checkpoint1",
                             "font.family": "DejaVu Sans", "font.size": 10})

    def _record(self, path, section, caption, caveat, source):
        self.records.append({"file": path.relative_to(self.folder).as_posix(),
                             "notebook_section": section, "caption": caption,
                             "interpretation_limit": caveat, "source": source})

    def figure(self, fig, name, section, caption, caveat="", source="results/checkpoint1/{linear,lora,ssf}/: result.json, history.csv and validation_last.npz"):
        for extension in ("png", "svg"):
            path = self.folder / "figures" / f"{name}.{extension}"
            metadata = {"Software": "PEFT Checkpoint 1"} if extension == "png" else {
                "Date": None, "Creator": "PEFT Checkpoint 1"}
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            self._record(path, section, caption, caveat, source)
        return fig

    def table(self, frame, name, section, caption, caveat="", source="results/checkpoint1/{linear,lora,ssf}/: result.json, history.csv and validation_last.npz"):
        path = self.folder / "tables" / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")
        self._record(path, section, caption, caveat, source)
        return frame

    def diagram(self, name, section, caption, caveat="", source="Mermaid source"):
        manifest = self.folder / "diagrams/rendered.json"
        try:
            rendered = json.loads(manifest.read_text())
            item = rendered["diagrams"][name]
            config = self.folder / "diagrams/mermaid-config.json"
            checks = [(config, rendered["config_sha256"])]
            checks.extend((self.folder / item[k], item[k + "_sha256"]) for k in ("source", "png", "svg"))
            if any(not path.is_file() or sha256(path) != expected for path, expected in checks):
                raise ValueError("Source or rendered file changed")
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Re-render Mermaid diagram '{name}' with scripts/render_diagrams.py") from exc
        with Image.open(self.folder / item["png"]) as image:
            image.verify()
        evidence = f"{source}, diagrams/{name}.mmd, Mermaid CLI {rendered['renderer']['version']}"
        for key in ("png", "svg", "source"):
            self._record(self.folder / item[key], section, caption, caveat, evidence)
        return self.folder / item["png"]

    def finalize(self):
        for row in self.records:
            path = self.folder / row["file"]
            assert path.is_file() and path.stat().st_size > 0
            row["sha256"] = sha256(path)
        index = pd.DataFrame(self.records)
        index.to_csv(self.folder / "asset_index.csv", index=False, lineterminator="\n")
        payload = {"scope": "Linear, LoRA and SSF, seed 42, epoch 100",
                   "notebook": "notebooks/01_flowers102_checkpoint1.ipynb",
                   "artifacts": self.records}
        (self.folder / "asset_index.json").write_text(json.dumps(payload, indent=2) + "\n")
        return index


def dataset_summary(snapshot):
    rows = []
    for split, frame in snapshot["records"].items():
        counts = frame["label"].value_counts()
        rows.append({"Split": split, "Gambar": len(frame), "Kelas teramati": len(counts),
                     "Kelas tanpa sampel": 102 - len(counts),
                     "Minimum gambar per kelas teramati": int(counts.min()),
                     "Maksimum gambar per kelas": int(counts.max())})
    return pd.DataFrame(rows)


def plot_dataset_examples(snapshot):
    photos = [x for x in snapshot["audit"]["images"] if x["role"] == "example"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, item in zip(axes.flat, photos):
        with Image.open(snapshot["folder"] / item["file"]) as photo:
            ax.imshow(ImageOps.pad(photo.convert("RGB"), (480, 320), color="white"))
        ax.set_title(f"Train baris {item['row']} | label {item['label']}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Contoh gambar training: delapan kelas berbeda", fontsize=15)
    fig.text(.5, .01, DATASET_CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    return fig


def plot_class_coverage(snapshot):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summary = dataset_summary(snapshot)
    colors = [COLORS["Linear"], COLORS["SSF"]]
    bars = axes[0].bar(summary["Split"], summary["Kelas teramati"], color=colors)
    axes[0].bar_label(bars, fmt="%d", padding=3)
    axes[0].axhline(102, color="#666666", linestyle="--", label="102 kelas model")
    axes[0].set(title="Cakupan kelas per split", ylabel="Jumlah kelas", ylim=(0, 115))
    axes[0].legend(fontsize=9)
    all_classes = np.arange(102)
    bins = np.arange(-.5, 11.6, 1)
    for (split, frame), color in zip(snapshot["records"].items(), colors):
        counts = frame["label"].value_counts().reindex(all_classes, fill_value=0)
        axes[1].hist(counts, bins=bins, histtype="step", linewidth=2, label=split, color=color)
    axes[1].set(title="Gambar per kelas, termasuk kelas tanpa sampel",
                xlabel="Gambar per kelas", ylabel="Jumlah kelas", xticks=range(0, 12))
    axes[1].legend()
    fig.tight_layout()
    return fig


def log_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Logits must be a finite two-dimensional array")
    return values - logsumexp(values, axis=1, keepdims=True)


def wilson_interval(correct, count, confidence=.95):
    if not 0 <= correct <= count or count <= 0:
        raise ValueError("Expected 0 <= correct <= positive count")
    z = NormalDist().inv_cdf(.5 + confidence / 2)
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    low = 0. if correct == 0 else max(0., center - half)
    high = 1. if correct == count else min(1., center + half)
    return low, high


def probability_metrics(probabilities, targets, log_probs=None):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets)
    if p.ndim != 2 or y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Targets must be one integer class index per probability row")
    if not len(p) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Probability rows must be finite, nonnegative and sum to one")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Target outside model class range")
    lp = np.log(np.clip(p, 1e-300, 1)) if log_probs is None else log_probs
    predicted = p.argmax(1)
    correct = int((predicted == y).sum())
    low, high = wilson_interval(correct, len(y))
    brier = np.mean(np.square(p).sum(1) - 2 * p[np.arange(len(y)), y] + 1)
    recalls = per_class_recall(predicted, y, p.shape[1])
    return {"accuracy_percent": 100 * correct / len(y), "correct": correct,
            "wrong": len(y) - correct, "n": len(y),
            "nll": float(-lp[np.arange(len(y)), y].mean()),
            "brier_multiclass_sum": float(max(0., brier)),
            "macro_recall_observed_percent": 100 * float(recalls["recall"].mean()),
            "classes_observed": int((recalls["validation_support"] > 0).sum()),
            "wilson95_low_percent": 100 * low, "wilson95_high_percent": 100 * high}


def per_class_recall(predicted, targets, class_count):
    predicted, targets = np.asarray(predicted), np.asarray(targets)
    if not isinstance(class_count, int) or class_count < 1:
        raise ValueError("Class count must be a positive integer")
    if predicted.shape != targets.shape or targets.ndim != 1 or not len(targets):
        raise ValueError("Expected matching nonempty prediction and target vectors")
    for values in (predicted, targets):
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= class_count).any():
            raise ValueError("Class indices must be integers inside the model class range")
    support = np.bincount(targets, minlength=class_count)
    correct = np.bincount(targets[predicted == targets], minlength=class_count)
    recall = np.divide(correct, support, out=np.full(class_count, np.nan), where=support > 0)
    return pd.DataFrame({"class_id": np.arange(class_count), "validation_support": support,
                         "correct": correct, "recall": recall})


def class_recall_table(runs, snapshot):
    frames = []
    training = snapshot["records"]["train"]["label"].value_counts()
    for name, run in runs.items():
        frame = per_class_recall(run["predicted"], run["targets"], run["logits"].shape[1])
        frame.insert(0, "method", name)
        frame["training_support"] = frame["class_id"].map(training).fillna(0).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def runtime_summary(history):
    seconds = np.asarray(history["epoch_seconds"], dtype=float)
    if seconds.ndim != 1 or not len(seconds) or not np.isfinite(seconds).all() or (seconds < 0).any():
        raise ValueError("Expected finite nonnegative durations for recorded epochs")
    return {"observed_total_epoch_minutes": float(seconds.sum() / 60),
            "observed_mean_epoch_seconds": float(seconds.mean())}


def load_completed_runs(root, snapshot):
    runs = {}
    expected_y = snapshot["records"]["validation"]["label"].to_numpy()
    for name in METHODS:
        folder = Path(root) / "results/checkpoint1" / name.lower()
        result = json.loads((folder / "result.json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        history = pd.read_csv(folder / "history.csv")
        assert result["phase"] == "complete" and result["epochs"] == 100
        assert result["seed"] == 42 and result["rows"] == {"train": 800, "validation": 200, "test_used": 0}
        assert result["frozen_backbone_unchanged"] and result["checkpoint_reload_exact"]
        assert history["epoch"].tolist() == list(range(1, 101))
        assert np.isclose(history["epoch_seconds"].sum(), result["total_epoch_seconds"], atol=1e-7, rtol=1e-10)
        for split, filename in (("train", "train800.txt"), ("validation", "val200.txt")):
            assert provenance["split_sha256"][filename] == snapshot["audit"]["splits"][split]["list_sha256"]
        with np.load(folder / "validation_last.npz") as stored:
            logits, y = stored["logits"].copy(), stored["targets"].copy()
        with np.load(folder / "validation_last_reloaded.npz") as reloaded:
            assert np.array_equal(logits, reloaded["logits"])
            assert np.array_equal(y, reloaded["targets"])
            reload_difference = float(np.max(np.abs(logits.astype(float) - reloaded["logits"].astype(float))))
        assert logits.shape == (200, 102) and np.array_equal(y, expected_y)
        lp = log_probabilities(logits)
        probabilities = np.exp(lp)
        scores = probability_metrics(probabilities, y, lp)
        assert scores["accuracy_percent"] == result["last_epoch_validation_top1_percent"]
        assert abs(scores["nll"] - result["final_validation_metrics"]["loss"]) <= .00501
        runs[name] = {"result": result, "provenance": provenance, "history": history,
                      "logits": logits, "targets": y, "log_probs": lp,
                      "probabilities": probabilities, "predicted": probabilities.argmax(1),
                      "reload_max_abs_logit_difference": reload_difference,
                      "metrics": scores, "folder": folder}
    reference = runs["Linear"]
    for run in runs.values():
        assert run["result"]["classifier_initial_sha256"] == reference["result"]["classifier_initial_sha256"]
        assert run["result"]["frozen_backbone_sha256"] == reference["result"]["frozen_backbone_sha256"]
    return runs


def reliability_bins(probabilities, targets, bins=5):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == targets
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        successes = int(correct[mask].sum())
        low, high = wilson_interval(successes, count) if count else (np.nan, np.nan)
        rows.append({"lower": edges[index], "upper": edges[index + 1], "count": count,
                     "mean_confidence": float(confidence[mask].mean()) if count else np.nan,
                     "accuracy": successes / count if count else np.nan,
                     "wilson95_low": low, "wilson95_high": high})
    return pd.DataFrame(rows)


def metrics_table(runs):
    rows = []
    for name, run in runs.items():
        result = run["result"]
        bins = reliability_bins(run["probabilities"], run["targets"])
        ece = (bins["count"] * (bins["accuracy"] - bins["mean_confidence"]).abs()).sum() / len(run["targets"])
        rows.append({"method": name, **run["metrics"], "ece_5_equal_width": float(ece),
                     "trainable_including_head": result["trainable_parameters_including_head"],
                     "adapter_parameters": result["trainable_parameters_excluding_head"],
                     "trainable_percent": result["trainable_percent"],
                     "best_validation_percent": result["best_validation_top1_percent"],
                     "best_validation_epoch": result["best_validation_epoch"],
                     **runtime_summary(run["history"]),
                     "observed_peak_gpu_allocated_mib": result["peak_gpu_allocated_mib"],
                     "observed_peak_process_rss_mib": result["peak_process_rss_mib"]})
    return pd.DataFrame(rows)


def prediction_comparisons(runs):
    disagreement = pd.DataFrame(0, index=METHODS, columns=METHODS)
    overlap = disagreement.copy()
    rows = []
    for a in METHODS:
        for b in METHODS:
            y = runs[a]["targets"]
            pa, pb = runs[a]["predicted"], runs[b]["predicted"]
            disagreement.loc[a, b] = int((pa != pb).sum())
            overlap.loc[a, b] = int(((pa != y) & (pb != y)).sum())
    for a, b in combinations(METHODS, 2):
        y = runs[a]["targets"]
        pa, pb = runs[a]["predicted"], runs[b]["predicted"]
        rows.append({"method_a": a, "method_b": b, "different_predictions": int((pa != pb).sum()),
                     "both_wrong": int(((pa != y) & (pb != y)).sum()),
                     "only_a_wrong": int(((pa != y) & (pb == y)).sum()),
                     "only_b_wrong": int(((pa == y) & (pb != y)).sum())})
    return disagreement, overlap, pd.DataFrame(rows)


def error_cases(runs, snapshot):
    y = runs["Linear"]["targets"]
    wrong = np.zeros(len(y), dtype=bool)
    for run in runs.values():
        wrong |= run["predicted"] != y
    counts = snapshot["records"]["train"]["label"].value_counts()
    photos = {item["row"]: item for item in snapshot["audit"]["images"] if item["role"] == "error_case"}
    rows = []
    for i in np.flatnonzero(wrong):
        number = int(i) + 1
        if number not in photos:
            raise FileNotFoundError(f"Missing photograph for validation error row {number}")
        row = {"validation_row": number, "true_label": int(y[i]),
               "training_examples_of_true_class": int(counts[y[i]]),
               "image_file": photos[number]["file"]}
        for name, run in runs.items():
            row[name + "_prediction"] = int(run["predicted"][i])
            row[name + "_confidence"] = float(run["probabilities"][i].max())
            row[name + "_correct"] = bool(run["predicted"][i] == y[i])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_predictions(logits, aggregation):
    if not logits:
        raise ValueError("At least one member is required")
    arrays = [np.asarray(x, dtype=float) for x in logits]
    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Ensemble members must have matching shapes and label order")
    if aggregation == "probability_mean":
        lp = logsumexp(np.stack([log_probabilities(x) for x in arrays]), axis=0) - np.log(len(arrays))
    elif aggregation == "logit_mean":
        lp = log_probabilities(np.mean(np.stack(arrays), axis=0))
    else:
        raise ValueError("Use probability_mean or logit_mean")
    return np.exp(lp), lp


def ensemble_table(runs):
    rows = []
    y = runs["Linear"]["targets"]
    for aggregation in ("probability_mean", "logit_mean"):
        for count in (2, 3):
            for members in combinations(METHODS, count):
                if not all(np.array_equal(y, runs[name]["targets"]) for name in members):
                    raise ValueError("Ensemble target order differs")
                average, lp = aggregate_predictions([runs[name]["logits"] for name in members], aggregation)
                rows.append({"ensemble": " + ".join(members), "members": count,
                             "aggregation": aggregation, **probability_metrics(average, y, lp),
                             "status": "exploratory_on_reused_validation"})
    return pd.DataFrame(rows)


def paired_nll_bootstrap(runs, resamples=10000, seed=42):
    if not isinstance(resamples, int) or resamples < 1 or len(runs) < 2:
        raise ValueError("Expected at least two runs and a positive resample count")
    reference_y = next(iter(runs.values()))["targets"]
    count = len(reference_y)
    if not count or any(not np.array_equal(reference_y, run["targets"]) for run in runs.values()):
        raise ValueError("Paired bootstrap requires the same nonempty target order")
    indices = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    losses = {name: -run["log_probs"][np.arange(count), reference_y] for name, run in runs.items()}
    rows = []
    for baseline, method in combinations(runs, 2):
        delta = losses[method] - losses[baseline]
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        rows.append({"method_a": method, "method_b": baseline,
                     "delta_nll_a_minus_b": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high),
                     "n": count, "resamples": resamples, "analysis_seed": seed})
    return pd.DataFrame(rows)


def provenance_table(runs):
    return pd.DataFrame([{
        "method": name, "epochs": run["result"]["epochs"],
        "classifier_initial_sha256": run["result"]["classifier_initial_sha256"],
        "frozen_backbone_sha256": run["result"]["frozen_backbone_sha256"],
        "recorded_backbone_unchanged": run["result"]["frozen_backbone_unchanged"],
        "target_order_verified": True,
        "reload_max_abs_logit_difference": run["reload_max_abs_logit_difference"],
    } for name, run in runs.items()])


def configuration_decisions(runs):
    config = runs["LoRA"]["result"]["config"]
    paper = "Mai et al., CVPR 2025, section 3 and Appendix A"
    default = "Pinned upstream experiment/config/default_vtab_processed.yml"
    rows = [
        ("Paper", "Backbone", "ViT-B/16, supervised ImageNet-21K", "Memakai backbone eksperimen VTAB paper dan menjaga perbandingan antarmetode.", "Belum mencakup backbone lain atau CLIP.", paper),
        ("Paper", "Optimizer", "AdamW", "Mengikuti protokol optimasi VTAB paper.", "Optimizer lain belum dibandingkan.", paper),
        ("Paper", "Epoch", str(config["epochs"]), "Anggaran training sama dan ditetapkan sebelum hasil diamati.", "Bukan bukti bahwa 100 epoch optimal untuk setiap metode.", paper),
        ("Paper", "Scheduler", "Cosine decay", "Mengikuti schedule upstream untuk menurunkan learning rate selama training.", "Schedule lain belum diablasi.", paper),
        ("Pilihan dalam grid", "Learning rate", str(config["learning_rate"]), "Pilihan awal dari grid VTAB [0.001, 0.01].", "Tidak menjalankan seluruh grid, sehingga bukan hasil tuning optimum.", "Paper Appendix A.1"),
        ("Pilihan dalam grid", "Weight decay", str(config["weight_decay"]), "Pilihan awal dari grid [0.0001, 0.001].", "Pengaruh strength regularisasi belum diisolasi.", "Paper Appendix A.1"),
        ("Pilihan dalam grid", "LoRA rank", str(config["lora_rank"]), "Rank 8 termasuk grid [1, 8, 16, 32] dengan kapasitas adaptasi kecil.", "Belum ada sweep rank pada eksperimen ini.", "Paper Table 3"),
        ("Default upstream", "LoRA dropout / scale", "0.1 / merge_factor=1", "Mempertahankan dropout input LoRA dan pengali delta pada kode native.", "Tidak memakai aturan alpha/r yang tidak terdapat pada implementasi ini.", "Pinned model/lora.py and default_vtab_processed.yml"),
        ("Paper", "Drop path", str(config["drop_path_rate"]), "Paper merekomendasikan regularisasi drop path nonzero dalam low-shot training.", "Manfaat khusus pada run kita belum diukur melalui ablation 0 versus 0.1.", "Paper section 3 and Figure 11"),
        ("Default upstream", "Warmup", f"{config['warmup_epochs']} epoch, LR awal {config['warmup_learning_rate']:g}", "Mempertahankan transisi learning rate dari default resmi.", "Lama warmup belum dituning.", default),
        ("Default upstream", "Minimum LR", f"{config['minimum_learning_rate']:g}", "Mempertahankan floor cosine schedule upstream.", "Floor lain belum dibandingkan.", default),
        ("Default upstream", "Transformasi", "Resize bicubic 224 x 224, ToTensor, normalized=False", "Mempertahankan transformasi dataset VTAB dari implementasi yang dipin.", "Efek normalization belum diablasi.", default + " and data/dataset/vtab_processed.py"),
        ("Paper", "Augmentation", "Tidak diaktifkan", "Paper menjaga konsistensi VTAB karena augmentation yang sama belum tentu menguntungkan 19 task.", "Kita belum menguji augmentation khusus Flowers102.", "Paper Appendix A.1"),
        ("Default upstream", "Seed", str(config["seed"]), "Mempertahankan seed default dan memeriksa classifier awal yang sama melalui hash.", "Satu seed tidak mengukur kestabilan hasil.", default),
        ("Resource", "Effective batch / microbatch", "64 / 8 untuk LoRA dan SSF, Linear batch langsung 64", "Accumulation mengurangi kebutuhan memori per microbatch dengan 12 update per epoch.", "Dropout draws dan penjumlahan numerik dapat berbeda dari batch langsung.", "Run configs and scripts/gradient_accumulation.py"),
        ("Resource", "Loader dan precision", "workers=0, pin_memory=False, satu thread, FP32", "Membatasi alokasi host pada node bersama dan menjaga satu precision selama baseline.", "Parallel loader atau mixed precision belum dibandingkan.", "scripts/train_validation.py and run provenance"),
        ("Protokol", "Drop last", "True, 768 gambar berkontribusi per epoch", "Mempertahankan loader upstream dengan effective batch yang utuh.", "32 dari 800 gambar tidak berkontribusi pada satu epoch, dengan shuffle setiap epoch.", "Native loader and train_validation.py"),
        ("Protokol", "Checkpoint dan resume", "Model terlatih, optimizer, scheduler, RNG dan history disimpan", "Melanjutkan run yang terputus resource tanpa mengulang dari nol.", "Sumber dan kebijakan resource ketika resume dicatat dalam provenance.", "scripts/train_validation.py"),
        ("Protokol", "Pemilihan hasil", "Validation epoch 100", "Aturan utama tetap meskipun checkpoint lebih awal memiliki skor tertinggi.", "Best checkpoint ditampilkan sebagai diagnostik.", "Protocol recorded before full runs"),
    ]
    return pd.DataFrame(rows, columns=["Kategori", "Pengaturan", "Nilai aktual", "Pertimbangan", "Konsekuensi", "Sumber"])


def paper_comparison():
    return pd.DataFrame([
        ("Lingkup", "19 task VTAB, full FT, linear dan 14 metode PEFT", "Flowers102, Linear, LoRA dan SSF"),
        ("Pemilihan konfigurasi", "Hyperparameter search pada 800 train / 200 validation", "Satu konfigurasi tetap, tanpa search"),
        ("Skor yang dilaporkan", "Retrain pada 1.000 gambar dan evaluasi original test", "Validation epoch 100 dari 800 train"),
        ("Backbone utama", "ViT-B/16 supervised ImageNet-21K", "ViT-B/16 supervised ImageNet-21K"),
        ("Error diversity analysis", "Caption menyebut majority vote, rincian Figure 4 Appendix C memakai average logits", "Error overlap serta probability averaging dan logits pada validation"),
        ("Dataset", "Processed VTAB dari arsip penulis", "Pinned mirror, exact membership terhadap arsip penulis belum diverifikasi"),
        ("Komputasi", "Memori dan FLOPs di luar fokus studi", "Resource tercatat sebagai diagnostik, tanpa benchmark terkontrol"),
    ], columns=["Aspek", "Paper", "Checkpoint 1"])
