"""CPU figures for the completed Flowers102 experiments."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps

from checkpoint_analysis import COLORS, METHODS, DATASET_CREDIT, reliability_bins


def learning_curves(runs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, run in runs.items():
        h = run["history"]
        color = COLORS[name]
        axes[0].plot(h["epoch"], h["val_top1"], color=color, label=name)
        axes[0].plot(h["epoch"], h["train_top1"], color=color, linestyle="--", alpha=.45)
        axes[1].plot(h["epoch"], h["val_top1"], color=color, label=name)
        axes[2].plot(h["epoch"], h["val_loss"], color=color, label=name)
    axes[0].set(title="Training dan validation", xlabel="Epoch", ylabel="Top-1 (%)", ylim=(0, 101))
    axes[1].set(title="Detail validation epoch 10-100", xlabel="Epoch", ylabel="Top-1 (%)",
                xlim=(10, 100), ylim=(96, 100.2))
    axes[2].set(title="Loss validation sepanjang training", xlabel="Epoch", ylabel="Cross-entropy")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.text(.5, .005, "Garis putus-putus: training. Loss history mengikuti pembulatan dua desimal dari runner.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def parameter_counts(summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    adapter = summary["adapter_parameters"].to_numpy() / 1000
    head = (summary["trainable_including_head"] - summary["adapter_parameters"]).to_numpy() / 1000
    ax.bar(summary["method"], head, label="Classifier head", color="#A8B4C8")
    bars = ax.bar(summary["method"], adapter, bottom=head, label="Parameter metode", color="#7356A6")
    ax.bar_label(bars, labels=[f"{v / 1000:.1f}k" for v in summary["trainable_including_head"]], padding=4)
    ax.set(title="Parameter yang dilatih, termasuk classifier", ylabel="Ribu parameter", ylim=(0, 450))
    ax.legend()
    fig.tight_layout()
    return fig


def final_accuracy(summary):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in summary.iterrows():
        score = row["accuracy_percent"]
        ax.errorbar(score, i, xerr=[[score - row["wilson95_low_percent"]],
                                  [row["wilson95_high_percent"] - score]],
                    fmt="o", capsize=5, color=COLORS[row["method"]], markersize=8)
        ax.text(score + .1, i + .15, f"{int(row['correct'])}/200", fontsize=10)
    ax.set(yticks=range(len(summary)), yticklabels=summary["method"],
           xlabel="Akurasi validation (%)", title="Epoch 100: akurasi dan interval Wilson 95%",
           xlim=(94, 100.7), ylim=(-.5, 2.6))
    ax.invert_yaxis()
    fig.text(.5, .005, "Interval binomial deskriptif untuk split ini. Satu gambar setara 0,5 poin persentase.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def probability_quality(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, field, title in zip(axes, ("nll", "brier_multiclass_sum"),
                                ("Negative log-likelihood", "Multiclass Brier score")):
        bars = ax.bar(summary["method"], summary[field], color=[COLORS[n] for n in summary["method"]])
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set(title=title, ylabel="Lebih rendah lebih baik", ylim=(0, summary[field].max() * 1.25))
    fig.suptitle("Akurasi sama, kualitas probabilitas berbeda", fontsize=14)
    fig.text(.5, .005, "Dihitung ulang dari logits epoch 100. Brier dijumlahkan atas 102 kelas, lalu dirata-ratakan atas gambar.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    return fig


def reliability(runs):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1.25]})
    for column, (name, run) in enumerate(runs.items()):
        frame = reliability_bins(run["probabilities"], run["targets"])
        valid = frame.loc[frame["count"] > 0]
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        ax.errorbar(valid["mean_confidence"], valid["accuracy"],
                    yerr=[valid["accuracy"] - valid["wilson95_low"],
                          valid["wilson95_high"] - valid["accuracy"]],
                    fmt="o", capsize=3, color=COLORS[name])
        ax.set(title=name, xlim=(0, 1.02), ylim=(0, 1.07), xlabel="Rata-rata confidence", ylabel="Akurasi bin")
        bottom = axes[1, column]
        bars = bottom.bar((frame["lower"] + frame["upper"]) / 2, frame["count"], width=.17, color=COLORS[name])
        bottom.bar_label(bars, fmt="%d", padding=2, fontsize=8)
        bottom.set(xlim=(0, 1), ylim=(0, 220), xlabel="Confidence, lima bin tetap", ylabel="Sampel")
    fig.suptitle("Reliability dan dukungan sampel per bin", fontsize=14)
    fig.text(.5, .005, "Whisker: Wilson 95% dalam bin. Bin kosong tidak menjadi titik. Estimasi terbatas oleh 200 gambar.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    return fig


def disagreement_matrices(disagreement, overlap):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, frame, title in zip(axes, (disagreement, overlap),
                                ("Prediksi berbeda dari 200 gambar", "Kedua metode salah pada gambar sama")):
        maximum = max(1, frame.to_numpy().max())
        ax.imshow(frame.to_numpy(), cmap="Blues", vmin=0, vmax=maximum)
        for i in range(3):
            for j in range(3):
                value = int(frame.iloc[i, j])
                ax.text(j, i, str(value), ha="center", va="center", fontsize=17,
                        color="white" if value > maximum / 2 else "#222222")
        ax.set(xticks=range(3), xticklabels=METHODS, yticks=range(3), yticklabels=METHODS, title=title)
        ax.grid(False)
    fig.tight_layout()
    return fig


def error_images(cases, snapshot):
    count = len(cases)
    fig, axes = plt.subplots(1, count, figsize=(3.3 * count, 5.8), squeeze=False)
    for ax, (_, case) in zip(axes.flat, cases.iterrows()):
        with Image.open(snapshot["folder"] / case["image_file"]) as photo:
            ax.imshow(ImageOps.pad(photo.convert("RGB"), (480, 360), color="white"))
        ax.set_title(f"Validation baris {case['validation_row']}\nLabel aktual: {case['true_label']}", fontsize=11)
        ax.axis("off")
        for j, name in enumerate(METHODS):
            correct = bool(case[name + "_correct"])
            text = f"{name}: {case[name + '_prediction']} ({case[name + '_confidence']:.1%})"
            ax.text(.02, -.10 - .13 * j, text, transform=ax.transAxes, fontsize=10,
                    color="#286B42" if correct else "#B33737")
    fig.suptitle("Lima gambar unik yang salah pada setidaknya satu metode", fontsize=15)
    fig.text(.5, .09, "Hijau: benar. Merah: salah. Persentase adalah confidence kelas prediksi.", ha="center", fontsize=9)
    fig.text(.5, .025, DATASET_CREDIT, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .26, 1, .90))
    return fig


def individual_error(case, snapshot):
    fig, ax = plt.subplots(figsize=(5, 5.7))
    with Image.open(snapshot["folder"] / case["image_file"]) as photo:
        ax.imshow(ImageOps.pad(photo.convert("RGB"), (480, 360), color="white"))
    ax.set_title(f"Validation baris {case['validation_row']} | aktual {case['true_label']}", fontsize=13)
    ax.axis("off")
    for j, name in enumerate(METHODS):
        correct = bool(case[name + "_correct"])
        ax.text(.05, -.08 - .11 * j,
                f"{name}: label {case[name + '_prediction']}, confidence {case[name + '_confidence']:.1%}",
                transform=ax.transAxes, fontsize=11, color="#286B42" if correct else "#B33737")
    fig.text(.5, .02, "Oxford Flowers102 via pinned VTAB-1K mirror", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .23, 1, 1))
    return fig


def ensemble_results(table):
    names = table["ensemble"].drop_duplicates().tolist()
    labels = [name.replace(" + ", "\n+ ") for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    settings = [("probability_mean", "Mean probabilitas", "#7356A6", -.15),
                ("logit_mean", "Mean logits", "#387A68", .15)]
    for key, label, color, offset in settings:
        subset = table.loc[table["aggregation"] == key].set_index("ensemble").loc[names]
        x = np.arange(len(names)) + offset
        score = subset["accuracy_percent"].to_numpy()
        axes[0].errorbar(x, score,
                        yerr=[score - subset["wilson95_low_percent"].to_numpy(),
                              subset["wilson95_high_percent"].to_numpy() - score],
                        fmt="o", color=color, capsize=3, markersize=6, label=label)
        for position, (_, row) in zip(x, subset.iterrows()):
            axes[0].annotate(f"{int(row['correct'])}/{int(row['n'])}", (position, row["accuracy_percent"]),
                             xytext=(0, 10 if offset < 0 else 23), textcoords="offset points",
                             ha="center", fontsize=8, color=color)
        bars = axes[1].bar(x, subset["nll"], width=.28, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axes[0].axhline(99, linestyle="--", color="#777777", linewidth=1)
    axes[0].set(xticks=range(len(names)), xticklabels=labels, ylabel="Akurasi validation (%)",
                ylim=(94, 102), title="Akurasi dan interval Wilson 95%")
    axes[1].set(xticks=range(len(names)), xticklabels=labels, ylabel="NLL, lebih rendah lebih baik",
                title="Dua aturan agregasi dengan bobot sama", ylim=(0, table["nll"].max() * 1.45))
    for ax in axes:
        ax.legend(loc="lower left" if ax is axes[0] else "upper right", fontsize=8)
    fig.suptitle("Eksplorasi ensemble pada validation yang sudah dianalisis", fontsize=14)
    fig.text(.5, .005, "Tidak ada bobot yang dioptimasi. Hasil ini belum menjadi estimasi performa test independen.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    return fig


def vit_backbone_diagram():
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set(xlim=(0, 13), ylim=(0, 2))
    ax.axis("off")
    labels = ["Gambar RGB\n224 x 224", "Patch 16 x 16\n196 patch", "Embedding + CLS\n197 token x 768",
              "12 blok Transformer\nattention + MLP", "Fitur CLS\n768 dimensi", "Classifier\n102 kelas"]
    for i, label in enumerate(labels):
        x = .05 + i * 2.16
        ax.add_patch(Rectangle((x, .55), 1.9, .9, facecolor="#EEF1F7", edgecolor="#465D86"))
        ax.text(x + .95, 1, label, ha="center", va="center", fontsize=10)
        if i < 5:
            ax.annotate("", xy=(x + 2.14, 1), xytext=(x + 1.92, 1),
                        arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_title("Alur ViT-B/16 untuk klasifikasi Flowers102", fontsize=14)
    fig.tight_layout()
    return fig


def experiment_flow():
    fig, ax = plt.subplots(figsize=(14, 2.6))
    ax.set(xlim=(0, 14), ylim=(0, 2.3))
    ax.axis("off")
    labels = ["Snapshot data\n800 train / 200 val", "ViT-B/16\nImageNet-21K",
              "Linear / LoRA / SSF\nbackbone frozen", "100 epoch\nseed 42",
              "Validation epoch 100\nlogits dan checkpoint", "Analisis\nmetrik dan error"]
    for i, label in enumerate(labels):
        x = .05 + i * 2.33
        ax.add_patch(Rectangle((x, .65), 2.05, .95, facecolor="#EEF1F7", edgecolor="#465D86", linewidth=1.2))
        ax.text(x + 1.025, 1.125, label, ha="center", va="center", fontsize=11)
        if i < 5:
            ax.annotate("", xy=(x + 2.29, 1.125), xytext=(x + 2.06, 1.125),
                        arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(7, .2, "Test tidak digunakan untuk training, pemilihan konfigurasi, atau evaluasi performa.",
            ha="center", fontsize=10)
    fig.tight_layout()
    return fig


def method_diagram():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set(xlim=(0, 11), ylim=(0, 6))
    ax.axis("off")
    labels = ["Linear", "LoRA", "SSF"]
    descriptions = ["", "A: 768 -> 8, B: 8 -> 2304\nDelta Q, K, V dalam 12 blok",
                    "Scale dan shift fitur\ngamma * z + beta"]
    for i, (name, description) in enumerate(zip(labels, descriptions)):
        y = 4.5 - i * 1.5
        ax.text(.05, y + .4, name, fontsize=13, fontweight="bold", va="center")
        ax.text(1.75, y + .4, "Gambar\n224 x 224", ha="center", va="center", fontsize=10)
        ax.add_patch(Rectangle((2.6, y - .1), 4.2, 1.1, facecolor="#E7EDF7", edgecolor="#465D86"))
        ax.text(4.7, y + .72, "ViT-B/16: bobot pretrained frozen", ha="center", fontsize=11)
        if description:
            ax.add_patch(Rectangle((2.85, y), 3.7, .48, facecolor="#FFF0D8", edgecolor="#C47723"))
            ax.text(4.7, y + .23, description, ha="center", va="center", fontsize=9.5)
        else:
            ax.text(4.7, y + .23, "Fitur pretrained", ha="center", va="center", fontsize=10)
        ax.add_patch(Rectangle((7.55, y + .05), 1.65, .75, facecolor="#FFF0D8", edgecolor="#C47723"))
        ax.text(8.375, y + .425, "Classifier\n78.438 parameter", ha="center", va="center", fontsize=10)
        ax.text(10.25, y + .4, "102 kelas", ha="center", va="center", fontsize=10)
        for start, end in ((2.2, 2.58), (6.82, 7.53), (9.22, 9.75)):
            ax.annotate("", xy=(end, y + .4), xytext=(start, y + .4),
                        arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(5.5, .25, "Biru: bobot frozen. Oranye: parameter yang dilatih. Skema mengikuti implementasi pinned.",
            ha="center", fontsize=10)
    fig.tight_layout()
    return fig


def research_direction_diagram():
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2))
    def box(ax, x, y, w, label):
        ax.add_patch(Rectangle((x, y), w, .65, facecolor="#EEF1F7", edgecolor="#465D86"))
        ax.text(x + w / 2, y + .325, label, ha="center", va="center", fontsize=10)
    def arrow(ax, start, end, label=""):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#555555"})
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .12,
                    label, ha="center", fontsize=8)
    for ax in axes:
        ax.set(xlim=(0, 12), ylim=(0, 2.2))
        ax.axis("off")
    ax = axes[0]
    ax.set_title("A. Fusion selektif berdasarkan confidence", loc="left", fontsize=13, fontweight="bold")
    box(ax, .1, .7, 1.7, "Linear")
    box(ax, 2.5, .7, 2.0, "Confidence\nterkalibrasi")
    box(ax, 5.6, .1, 1.6, "LoRA")
    box(ax, 8.3, .1, 2.6, "Fusion probabilitas")
    box(ax, 8.3, 1.35, 2.6, "Terima prediksi Linear")
    arrow(ax, (1.82, 1.025), (2.48, 1.025))
    arrow(ax, (4.52, 1.025), (5.58, .425), "kurang yakin")
    arrow(ax, (7.22, .425), (8.28, .425))
    arrow(ax, (4.52, 1.2), (8.28, 1.675), "yakin")
    ax = axes[1]
    ax.set_title("B. Pembobotan yang menjaga kontribusi kelas kecil", loc="left", fontsize=13, fontweight="bold")
    labels = ["Loss sampel\ntraining", "Bobot relatif\ndalam kelas", "Batas kontribusi\nminimum per kelas", "Training\nLoRA / SSF"]
    for i, label in enumerate(labels):
        x = .1 + i * 3
        box(ax, x, .65, 2.35, label)
        if i < 3:
            arrow(ax, (x + 2.37, .975), (x + 2.98, .975))
    ax = axes[2]
    ax.set_title("C. Pemilihan resep dengan anggaran resource", loc="left", fontsize=13, fontweight="bold")
    labels = ["Resep\nkandidat", "Profiling\nsingkat", "Filter kelayakan\nRAM / GPU", "Pilih dengan\nkualitas dan biaya", "Evaluasi\nindependen"]
    for i, label in enumerate(labels):
        x = .1 + i * 2.4
        box(ax, x, .65, 1.95, label)
        if i < 4:
            arrow(ax, (x + 1.97, .975), (x + 2.38, .975))
    fig.suptitle("Usulan kontribusi: mekanisme yang akan diuji", fontsize=15)
    fig.text(.5, .01, "Tidak ada hasil evaluasi gate, noise weighting, atau resource selector pada Checkpoint 1.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    return fig
