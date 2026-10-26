"""Regenerate every result chart of the report from the numbers in the drafting prompt
(Part B/D). Output: 'Results Figures/*.png'.   Usage: python scripts/make_figures.py
All values are typed in below so that the charts can be checked against the report tables.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent.parent / "Results Figures"
OUT.mkdir(exist_ok=True)

DT, RF = "#2b6cb0", "#dd6b20"          # Decision Tree blue, Random Forest orange
GREY = "#7a7a7a"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def comma(x, _=None):
    return f"{x:,.0f}" if x >= 1 else f"{x:g}"


# ---------------------------------------------------------------- 1. overall metrics
def fig_metrics():
    names = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced\naccuracy", "MCC"]
    dt = [0.9765, 0.9431, 0.9940, 0.9679, 0.9669, 0.9804, 0.9502]
    rf = [0.9964, 0.9954, 0.9946, 0.9950, 0.9974, 0.9960, 0.9922]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, dt, w, color=DT, label="Decision Tree (44 leaves)")
    b2 = ax.bar(x + w / 2, rf, w, color=RF, label="Random Forest (50 trees)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008, f"{b.get_height():.4f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.axhline(0.95, color=GREY, ls="--", lw=0.8, label="0.95 design target")
    ax.set_ylim(0.90, 1.006); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Score (y-axis starts at 0.90)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, "r_metrics.png")


# ---------------------------------------------------------------- 2. confusion matrices
def fig_confusion():
    cms = {"Decision Tree": np.array([[1975121, 67667], [6826, 1122020]]),
           "Random Forest": np.array([[2037559, 5229], [6104, 1122742]])}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, (name, cm) in zip(axes, cms.items()):
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({frac[i, j] * 100:.2f}% of row)", ha="center", va="center",
                        fontsize=7, color="white" if frac[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-IoT", "IoT"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["non-IoT", "IoT"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
        ax.grid(False)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "r_confusion.png")


# ---------------------------------------------------------------- 3. per day
def fig_perday():
    days = ["16-10-09", "16-10-10", "16-10-11", "16-10-12"]
    acc_dt = [0.9861, 0.9898, 0.9862, 0.9673]; acc_rf = [0.9901, 0.9929, 0.9950, 0.9989]
    f1_dt = [0.9927, 0.9947, 0.9839, 0.8685]; f1_rf = [0.9947, 0.9963, 0.9941, 0.9951]
    x = np.arange(4); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax, (a, b, t) in zip(axes, [(acc_dt, acc_rf, "Accuracy"), (f1_dt, f1_rf, "F1 (IoT)")]):
        ax.bar(x - w / 2, a, w, color=DT, label="Decision Tree")
        ax.bar(x + w / 2, b, w, color=RF, label="Random Forest")
        for xi, v in zip(x - w / 2, a):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=6, rotation=90, va="bottom")
        for xi, v in zip(x + w / 2, b):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=6, rotation=90, va="bottom")
        ax.set_ylim(0.85, 1.03); ax.set_xticks(x); ax.set_xticklabels(days, fontsize=7.5)
        ax.set_title(t + " (y-axis starts at 0.85)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, "r_perday.png")


# ---------------------------------------------------------------- 4. per device
def fig_perdevice():
    dev = [("00:16:6c:ab:6b:88", "IoT", 0.999969, 0.993673), ("00:24:e4:1b:6f:96", "IoT", 0.986928, 0.986928),
           ("00:24:e4:20:28:c6", "IoT", 0.999789, 0.999816), ("00:62:6e:51:27:2e", "IoT", 0.944502, 0.952891),
           ("08:21:ef:3b:fc:e3", "non-IoT", 0.886535, 0.920296), ("18:b4:30:25:be:e4", "IoT", 0.042553, 0.968085),
           ("18:b7:9e:02:20:44", "IoT", 0.991409, 0.991564), ("30:8c:fb:2f:e4:b2", "IoT", 0.999947, 0.999901),
           ("44:65:0d:56:cc:d3", "IoT", 0.993544, 0.998632), ("70:5a:0f:e4:9b:c0", "IoT", 0.992199, 0.994769),
           ("70:ee:50:03:b8:ac", "IoT", 1.0, 1.0), ("70:ee:50:18:34:43", "IoT", 0.988432, 0.993977),
           ("74:2f:68:81:69:42", "non-IoT", 0.969533, 0.999862), ("b4:ce:f6:a7:a3:c2", "non-IoT", 0.865379, 0.926523),
           ("d0:52:a8:00:67:5e", "IoT", 1.0, 0.999955), ("d0:73:d5:01:83:08", "IoT", 0.999878, 1.0),
           ("ec:1a:59:79:f4:89", "IoT", 0.999845, 0.999932), ("ec:1a:59:83:28:11", "IoT", 0.999303, 0.999226)]
    n = len(dev); y = np.arange(n)[::-1]; h = 0.4
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(y + h / 2, [d[2] for d in dev], h, color=DT, label="Decision Tree")
    ax.barh(y - h / 2, [d[3] for d in dev], h, color=RF, label="Random Forest")
    ax.set_yticks(y); ax.set_yticklabels([f"{d[0]} ({d[1]})" for d in dev], fontsize=7)
    ax.set_xlim(0, 1.02); ax.set_xlabel("Fraction of the device's test packets classified correctly")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "r_perdevice.png")


# ---------------------------------------------------------------- 5. CV curve (Decision Tree)
def fig_cv_alpha():
    alpha = [0, 5.45e-12, 2.50e-11, 1.64e-10, 8.85e-10, 4.34e-09, 2.47e-08, 1.37e-07, 7.40e-07,
             4.01e-06, 2.16e-05, 1.19e-04, 6.61e-04, 3.59e-03, 1.66e-02, 1.01e-01]
    f1 = [0.984155, 0.984155, 0.984155, 0.984155, 0.984155, 0.984155, 0.984157, 0.984163, 0.984201,
          0.974207, 0.975822, 0.982085, 0.976058, 0.946610, 0.917164, 0.851405]
    sd = [0.009470, 0.009470, 0.009470, 0.009470, 0.009470, 0.009470, 0.009468, 0.009473, 0.009653,
          0.020414, 0.021394, 0.006912, 0.015378, 0.032980, 0.064092, 0.048300]
    a = np.array(alpha, dtype=float); a_plot = np.where(a == 0, 1e-12, a)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.errorbar(a_plot, f1, yerr=sd, fmt="o-", color=DT, ms=4, lw=1, capsize=2, elinewidth=0.7)
    thr = 0.984201 - 0.009653
    ax.axhline(thr, color=GREY, ls="--", lw=0.8)
    ax.text(1.2e-12, thr - 0.012, f"selection threshold {thr:.5f}", fontsize=7, color=GREY)
    ax.plot([7.4e-7], [0.984201], "s", color="green", ms=7, label="best CV F1 (alpha 7.4e-07)")
    ax.plot([6.61e-4], [0.976058], "D", color=RF, ms=7, label="selected (alpha 6.6e-04)")
    ax.set_xscale("log"); ax.set_xlabel("ccp_alpha (alpha = 0 drawn at 1e-12)")
    ax.set_ylabel("Mean CV F1 (+/- 1 SD over 5 folds)"); ax.set_ylim(0.78, 1.03)
    ax.legend(loc="lower left", frameon=False)
    save(fig, "r_cv_alpha.png")


# ---------------------------------------------------------------- 6. sweep: F1 vs leaves
SWEEP = [  # leaves, test F1, us/packet (batch), label
    (5784, 0.99780, 0.0745), (5750, 0.99780, 0.0735), (5641, 0.99780, 0.0708), (5192, 0.99780, 0.0673),
    (1257, 0.99722, 0.0621), (211, 0.99464, 0.0609), (44, 0.96787, 0.0472), (2, 0.81689, 0.0393)]


def fig_sweep_leaves():
    L = [s[0] for s in SWEEP]; F = [s[1] for s in SWEEP]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(L, F, "o-", color=DT, ms=4)
    for l, f in zip(L, F):
        if l in (5784, 1257, 211, 44, 2):
            ax.annotate(f"{l:,} leaves\nF1 {f:.4f}", (l, f), textcoords="offset points", xytext=(6, 8 if l == 2 else (-22 if l < 100 else 6)),
                        fontsize=6.5)
    ax.plot([44], [0.96787], "D", color=RF, ms=8, label="selected tree (44 leaves)")
    ax.axhline(0.9950, color=RF, ls="--", lw=0.9, label="Random Forest F1 0.9950")
    ax.set_xscale("log"); ax.set_xlabel("Number of leaves (log scale)"); ax.set_ylabel("Test F1 (IoT)")
    ax.set_ylim(0.80, 1.005)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "r_sweep_leaves.png")


# ---------------------------------------------------------------- 7. trade-off F1 vs us/packet
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot([s[2] for s in SWEEP[:-1]], [s[1] for s in SWEEP[:-1]], "o-", color=DT, ms=4,
            label="Decision Tree, post-hoc pruning sweep")
    rf = [(1, 0.064, 0.98301), (5, 0.268, 0.99122), (10, 0.533, 0.99509), (25, 1.233, 0.99513), (50, 2.554, 0.99498)]
    ax.plot([r[1] for r in rf], [r[2] for r in rf], "s-", color=RF, ms=4, label="Random Forest, 1 to 50 trees")
    for n, t, f in rf:
        ax.annotate(f"{n} tree" + ("s" if n > 1 else ""), (t, f), textcoords="offset points", xytext=(4, -10), fontsize=6.5)
    for l, f, t in SWEEP[:-1]:
        if l in (1257, 211, 44):
            ax.annotate(f"{l:,} leaves", (t, f), textcoords="offset points", xytext={1257: (6, 4), 211: (-52, -6), 44: (-8, -12)}[l], fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("Batch inference time (microseconds per packet, log scale)")
    ax.set_ylabel("Test F1 (IoT), post-hoc, test set"); ax.set_ylim(0.955, 1.002)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "r_tradeoff.png")


# ---------------------------------------------------------------- 8. latency
def fig_latency():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    labs = ["p50", "p95", "p99"]; dt = [76.0, 152.8, 220.8]; rf = [2223.9, 2583.9, 3194.9]
    x = np.arange(3); w = 0.38
    b1 = ax.bar(x - w / 2, dt, w, color=DT, label="Decision Tree"); b2 = ax.bar(x + w / 2, rf, w, color=RF, label="Random Forest")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{b.get_height():,.1f}", ha="center", fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylim(10, 12000); ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("microseconds per call (log)"); ax.set_title("One packet per call")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    v = [0.0780, 2.6966]
    bars = ax.bar(["Decision\nTree", "Random\nForest"], v, color=[DT, RF], width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.1, f"{b.get_height():.4f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(0.01, 12); ax.set_ylabel("microseconds per packet (log)")
    ax.set_title("100,000 packets per call")
    save(fig, "r_latency.png")


# ---------------------------------------------------------------- 9. latency vs batch size
def fig_batch():
    b = [1, 10, 100, 1000, 10000, 100000]
    dt = [358.2, 43.94, 5.609, 0.2308, 0.0635, 0.0780]
    rf = [3552.7, 253.71, 29.006, 7.640, 2.877, 2.697]
    rfall = [36620.9, 3550.4, 347.3, 31.3, 3.12, 0.808]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(b, dt, "o-", color=DT, label="Decision Tree")
    ax.plot(b, rf, "s-", color=RF, label="Random Forest, one thread")
    ax.plot(b, rfall, "^--", color="#7b5ea7", label="Random Forest, all cores (n_jobs = -1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Packets per predict call"); ax.set_ylabel("Microseconds per packet (median of 5)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "r_batch.png")


# ---------------------------------------------------------------- 10. model size
def fig_modelsize():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    v = [8.1, 5693.3]
    bars = ax.bar(["Decision Tree\n(87 nodes)", "Random Forest\n(72,612 nodes)"], v, color=[DT, RF], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, f"{b.get_height():,.1f} KB", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1, 40000); ax.set_ylabel("Model file size (KB, log scale)")
    save(fig, "r_modelsize.png")


# ---------------------------------------------------------------- 11. training cost
def fig_training():
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = np.arange(2); w = 0.36
    wall = [447.8, 676.0]; cpu = [440.9, 6356.0]
    b1 = ax.bar(x - w / 2, wall, w, color="#9ab7d3", label="Wall-clock (s)")
    b2 = ax.bar(x + w / 2, cpu, w, color="#d98c5f", label="CPU-seconds")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.1, f"{b.get_height():,.1f}", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(100, 30000); ax.set_xticks(x); ax.set_xticklabels(["Decision Tree", "Random Forest"])
    ax.set_ylabel("Seconds (log scale)"); ax.legend(frameon=False, loc="upper left")
    save(fig, "r_training.png")


# ---------------------------------------------------------------- 12. training time vs data
def fig_trainscale():
    rows = [690222, 2070667, 6902224]
    dt = [2.9, 11.9, 42.7]; rf1 = [23.7, 100.8, 361.6]; rfa = [6.3, 32.7, 130.1]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(rows, dt, "o-", color=DT, label="Decision Tree")
    ax.plot(rows, rf1, "s-", color=RF, label="Random Forest, one thread")
    ax.plot(rows, rfa, "^--", color="#7b5ea7", label="Random Forest, all cores")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(rows); ax.set_xticklabels(["690,222\n(10%)", "2,070,667\n(30%)", "6,902,224\n(100%)"])
    ax.minorticks_off()
    ax.set_xlabel("Training rows"); ax.set_ylabel("Wall-clock time (s, mean)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "r_trainscale.png")


# ---------------------------------------------------------------- 13. CPU time per packet
def fig_cpu():
    ppc = ["1", "1,000", "100,000"]
    med = {"dt": [74.62, 0.1250, 0.04659], "rf": [2159.8, 5.037, 2.188]}
    q25 = {"dt": [74.22, 0.1245, 0.04648], "rf": [2149.5, 4.988, 2.177]}
    q75 = {"dt": [75.50, 0.1262, 0.04687], "rf": [2174.4, 5.044, 2.198]}
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for k, col, lab, off in (("dt", DT, "Decision Tree", -w / 2), ("rf", RF, "Random Forest", w / 2)):
        m = np.array(med[k]); lo = m - np.array(q25[k]); hi = np.array(q75[k]) - m
        bars = ax.bar(x + off, m, w, color=col, label=lab, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:g}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(0.01, 8000)
    ax.set_xticks(x); ax.set_xticklabels(ppc)
    ax.set_xlabel("Packets per predict call"); ax.set_ylabel("CPU time per packet (microseconds, log)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "r_cpu.png")


# ---------------------------------------------------------------- 14. load on one core
def fig_load():
    rates = [100, 1000, 10000, 100000]
    series = {("Decision Tree", 1): [0.746, 7.46, 74.6, 746], ("Decision Tree", 1000): [0.00125, 0.0125, 0.125, 1.25],
              ("Decision Tree", 100000): [0.000466, 0.00466, 0.0466, 0.466],
              ("Random Forest", 1): [21.6, 216, 2160, 21598], ("Random Forest", 1000): [0.0504, 0.504, 5.04, 50.4],
              ("Random Forest", 100000): [0.0219, 0.219, 2.19, 21.9]}
    ls = {1: "-", 1000: "--", 100000: ":"}; fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (m, b), v in series.items():
        ax.plot(rates, v, ls[b], marker="o", ms=3.5, color=DT if m == "Decision Tree" else RF,
                label=f"{m}, {b:,} per call")
    ax.axhline(100, color="red", lw=1); ax.text(105, 135, "100% = one core fully busy", color="red", fontsize=7, ha="left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Packet rate (packets per second)"); ax.set_ylabel("Calculated load on one core (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    save(fig, "r_load.png")


# ---------------------------------------------------------------- 15. memory added
def fig_memory():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    x = np.arange(3); w = 0.38; lab = ["1", "1,000", "100,000"]
    load_dt = [0.30, 0.31, 0.30]; load_rf = [11.51, 11.52, 11.52]
    gr_dt = [-2.30, -2.80, -1.12]; gr_rf = [-7.79, -8.16, -4.68]
    for ax, (a, b, t) in zip(axes, [(load_dt, load_rf, "Memory added by loading the model"),
                                    (gr_dt, gr_rf, "Resident-memory change during prediction")]):
        b1 = ax.bar(x - w / 2, a, w, color=DT, label="Decision Tree"); b2 = ax.bar(x + w / 2, b, w, color=RF, label="Random Forest")
        for bar in list(b1) + list(b2):
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("Packets per call"); ax.set_title(t, fontsize=8.5)
        ax.axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("MiB"); axes[0].set_ylim(0, 14.5); axes[1].set_ylim(-9.8, 1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.35)
    save(fig, "r_memory.png")


# ---------------------------------------------------------------- 16. peak composition
def fig_peakstack():
    comp = ["Python + NumPy (start)", "scikit-learn import", "Model", "Input batch", "Transient during prediction"]
    dt = [41.2, 120.6, 0.3, 8.8, 1.6]; rf = [41.1, 120.6, 11.5, 9.1, 3.1]
    cols = ["#8da0cb", "#66c2a5", "#e78ac3", "#ffd92f", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (name, d, r) in enumerate(zip(comp, dt, rf)):
        left = [sum(dt[:i]), sum(rf[:i])]
        ax.barh(["Decision Tree", "Random Forest"], [d, r], left=left, color=cols[i], label=name, height=0.5)
    ax.text(sum(dt) + 2, 0, "172.6 MiB (peak)", va="center", fontsize=8)
    ax.text(sum(rf) + 2, 1, "185.4 MiB (peak)", va="center", fontsize=8)
    ax.set_xlim(0, 235); ax.set_xlabel("Peak resident memory of the whole process (MiB), 100,000 packets per call")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    save(fig, "r_peakstack.png")


# ---------------------------------------------------------------- 17. test pass/fail summary (Chapter 6)
def fig_passfail():
    cats = ["Pass", "Mixed\n(one model passes,\nthe other fails)", "Error cases\npresent", "Not verified\nor not re-run"]
    n = [15, 3, 2, 2]; cols = ["#4caf50", "#f5b041", "#e57373", "#9e9e9e"]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    bars = ax.bar(cats, n, color=cols, width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, str(int(b.get_height())), ha="center", fontsize=9)
    ax.set_ylabel("Number of test cases (22 in total)"); ax.set_ylim(0, 18); ax.tick_params(axis="x", labelsize=7.5)
    save(fig, "r_passfail.png")


# ---------------------------------------------------------------- 18. cost split (Chapter 8)
def fig_costsplit():
    labour, hw, other = 502000, 6500, 21802
    cont = round(0.10 * (labour + hw + other))
    vals = [labour, other, hw, cont]; names = ["Labour (notional)", "Other costs", "Hardware", "Contingency (10%)"]
    cols = ["#4a78b5", "#8fb8de", "#6aa84f", "#e8a838"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.pie(vals[:1] + vals[1:], colors=cols, startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Share of total (INR 583,332)", fontsize=8.5)
    ax.legend(names, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    ax = axes[1]
    ax.barh(names[::-1], [v / 1000 for v in vals[::-1]], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v / 1000 + 8, i, f"{v:,}", va="center", fontsize=7.5)
    ax.set_xlim(0, 640); ax.set_xlabel("Thousand INR"); ax.grid(False)
    ax.tick_params(axis="y", labelsize=7.5)
    save(fig, "r_costsplit.png")


if __name__ == "__main__":
    for f in (fig_metrics, fig_confusion, fig_perday, fig_perdevice, fig_cv_alpha, fig_sweep_leaves, fig_tradeoff,
              fig_latency, fig_batch, fig_modelsize, fig_training, fig_trainscale, fig_cpu, fig_load, fig_memory,
              fig_peakstack, fig_passfail, fig_costsplit):
        f()
