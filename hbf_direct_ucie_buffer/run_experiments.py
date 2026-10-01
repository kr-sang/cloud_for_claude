"""Run the latency-hiding buffer experiments and emit CSV + PNG + markdown tables.

Usage:  python3 run_experiments.py [--quick]
Outputs go to ./results/
"""
from __future__ import annotations

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import (KiB, MiB, HBF_CHANNELS, HBF_CH_EFF_BW, HBF_MOCS, HBM_CHANNELS,
                 HBM_CH_BW, hbf_channel, hbm_channel, simulate, us, ns)

QUICK = "--quick" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# --- reference GPU on-chip capacities [external, not from the OCP spec] ------
L2_REFS = {
    "H100 L2 (50 MB)": 50e6,
    "MI300X Infinity Cache (256 MB)": 256e6,
}

# validated categorical palette (dataviz reference instance, light mode)
C = {"hbm": "#2a78d6", "hbf3": "#eb6834", "hbf20": "#1baf7a", "hbf50": "#eda100", "x": "#e87ba4"}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e6e5df",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "font.size": 10, "lines.linewidth": 2,
    "font.family": ["NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False,
})

NONLIMITING_BANKS = 4096   # sense units/channel large enough that the array never limits


def little(bw, lat):
    return bw * lat


def device_latency(cfg, req_bytes):
    """unloaded miss latency seen by the host for one request."""
    return cfg.t_cmd + cfg.t_sense + req_bytes / cfg.link_bw + cfg.t_resp


def sweep(cfg, req_bytes, windows_ch, warmup, measure):
    rows = []
    for W in windows_ch:
        t0 = time.time()
        r = simulate(cfg, W, req_bytes, warmup, measure)
        rows.append(r)
        print(f"  {cfg.name:28s} req={req_bytes:5d}B W/ch={W/KiB:8.0f}KiB -> "
              f"{r.achieved_bw/1e9:6.1f} GB/s/ch  lat={r.mean_latency/us:6.2f}us "
              f"out={r.mean_outstanding_reqs:7.0f} ({time.time()-t0:.1f}s)", flush=True)
    return rows


def knee(rows, cap, frac):
    """smallest window whose achieved BW >= frac*cap (None if never)."""
    for r in rows:
        if r.achieved_bw >= frac * cap:
            return r.window_bytes
    return None


def fmt_mb(b):
    return "-" if b is None else f"{b/1e6:.1f} MB"


results_md = []

# ============================================================================
# E1: achieved BW vs buffer size, 4 KiB requests, array non-limiting
# ============================================================================
print("E1: BW vs buffer")
warm_hbf = 150 * us if QUICK else 300 * us
meas_hbf = 300 * us if QUICK else 600 * us
devices = [
    ("HBM3e x6 (RT~0.55us)", hbm_channel(), HBM_CHANNELS, HBM_CH_BW, C["hbm"], 128,
     np.array([2, 4, 8, 12, 16, 24, 32, 48, 64, 128]) * KiB, 30 * us, 100 * us),
    ("HBF tR=3us (SLC class)", hbf_channel(3.0, NONLIMITING_BANKS), HBF_CHANNELS, HBF_CH_EFF_BW, C["hbf3"], 4 * KiB,
     np.array([64, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 4096]) * KiB, warm_hbf, meas_hbf),
    ("HBF tR=20us (commodity)", hbf_channel(20.0, NONLIMITING_BANKS), HBF_CHANNELS, HBF_CH_EFF_BW, C["hbf20"], 4 * KiB,
     np.array([256, 512, 1024, 2048, 3072, 3584, 4096, 4608, 5120, 6144, 8192, 16384]) * KiB, warm_hbf, meas_hbf),
    ("HBF tR=50us (TLC class)", hbf_channel(50.0, NONLIMITING_BANKS), HBF_CHANNELS, HBF_CH_EFF_BW, C["hbf50"], 4 * KiB,
     np.array([1024, 2048, 4096, 6144, 8192, 9216, 10240, 11264, 12288, 16384, 32768]) * KiB, warm_hbf, meas_hbf),
]
e1 = {}
with open(os.path.join(OUT, "e1_bw_vs_buffer.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["device", "req_bytes", "window_per_channel_B", "window_total_B", "achieved_bw_total_GBs",
                "cap_total_GBs", "mean_latency_us", "p99_latency_us", "mean_outstanding_reqs_per_ch"])
    for name, cfg, nch, capch, col, req, wins, warm, meas in devices:
        rows = sweep(cfg, req, wins, warm, meas)
        e1[name] = (rows, nch, capch, col, cfg, req)
        for r in rows:
            w.writerow([name, req, r.window_bytes, r.window_bytes * nch, r.achieved_bw * nch / 1e9,
                        capch * nch / 1e9, r.mean_latency / us, r.p99_latency / us, r.mean_outstanding_reqs])

fig, ax = plt.subplots(figsize=(9, 5.2))
table = ["| 장치 | 유효 BW 상한 | 무부하 miss latency | Little 추정 (BW×lat) | 시뮬 95% 도달 버퍼 | 시뮬 99% 도달 버퍼 | H100 L2 50 MB 대비 (99%) |",
         "|---|---|---|---|---|---|---|"]
for name, (rows, nch, capch, col, cfg, req) in e1.items():
    x = np.array([r.window_bytes * nch for r in rows]) / 1e6
    y = np.array([r.achieved_bw * nch for r in rows]) / 1e12
    ax.plot(x, y, marker="o", ms=5, color=col, label=name)
    lat = device_latency(cfg, req)
    L = little(capch, lat) * nch
    k95 = knee(rows, capch, 0.95); k99 = knee(rows, capch, 0.99)
    k95t = None if k95 is None else k95 * nch
    k99t = None if k99 is None else k99 * nch
    table.append(f"| {name} | {capch*nch/1e12:.2f} TB/s | {lat/us:.2f} µs | {L/1e6:.1f} MB | {fmt_mb(k95t)} | {fmt_mb(k99t)} | "
                 f"{'-' if k99t is None else f'{100*k99t/50e6:.0f}%'} |")
for lbl, v in L2_REFS.items():
    ax.axvline(v / 1e6, color="#c3c2b7", ls="--", lw=1)
    ax.text(v / 1e6 * 1.04, 0.15, lbl, rotation=90, fontsize=8, color="#52514e", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("GPU 측 in-flight 버퍼 (총량, MB, log)")
ax.set_ylabel("달성 대역폭 (TB/s)")
ax.set_title("E1. 버퍼 크기 vs 달성 대역폭 — 4 KiB 요청, NAND array 비제한 (link만 제한)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "e1_bw_vs_buffer.png"), dpi=150); plt.close(fig)
results_md.append(("E1. 버퍼 크기 vs 달성 대역폭 (4 KiB 요청)", "\n".join(table)))
print("\n".join(table))

# ============================================================================
# E2: request granularity (64 B L2-line misses vs 4 KiB bursts), MOCS on/off
# ============================================================================
print("E2: granularity")
e2_rows = []
table = ["| tR | 요청 크기 | MOCS 16,384/ch | 버퍼/채널 | 달성 BW (cube) | 평균 outstanding 요청/ch | 필요 outstanding 요청 (cube, Little) |",
         "|---|---|---|---|---|---|---|"]
grans = [64, 256, 1024, 4096]
for tR in ([3.0, 20.0] if QUICK else [3.0, 20.0, 50.0]):
    warm = max(150 * us, 4 * tR * us); meas = 300 * us
    for mocs in [HBF_MOCS, None]:
        cfg = hbf_channel(tR, NONLIMITING_BANKS, mocs=mocs)
        lat = device_latency(cfg, 64)
        Wch = int(little(HBF_CH_EFF_BW, lat) * 1.6 // (4 * KiB) + 1) * 4 * KiB  # 1.6x Little, ample
        for g in grans:
            if g == 64 and tR == 50.0 and not QUICK:
                meas_g = 150 * us
            else:
                meas_g = meas
            r = simulate(cfg, Wch, g, warm, meas_g)
            need = little(HBF_CH_EFF_BW, device_latency(cfg, g)) / g * HBF_CHANNELS
            e2_rows.append((tR, g, mocs, Wch, r))
            print(f"  tR={tR:4.0f}us g={g:5d} mocs={mocs} W/ch={Wch/KiB:.0f}KiB -> {r.achieved_bw*HBF_CHANNELS/1e9:7.0f} GB/s out={r.mean_outstanding_reqs:.0f}", flush=True)
            table.append(f"| {tR:g} µs | {g} B | {'on' if mocs else 'off'} | {Wch/MiB:.1f} MiB | {r.achieved_bw*HBF_CHANNELS/1e9:.0f} GB/s | "
                         f"{r.mean_outstanding_reqs:.0f} | {need:,.0f} |")
with open(os.path.join(OUT, "e2_granularity.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tR_us", "req_bytes", "mocs_per_ch", "window_per_ch_B", "achieved_bw_cube_GBs", "mean_outstanding_reqs_per_ch", "mean_latency_us"])
    for tR, g, mocs, Wch, r in e2_rows:
        w.writerow([tR, g, mocs or 0, Wch, r.achieved_bw * HBF_CHANNELS / 1e9, r.mean_outstanding_reqs, r.mean_latency / us])
results_md.append(("E2. 요청 입도와 MOCS(최대 outstanding 명령 수)의 영향", "\n".join(table)))

fig, ax = plt.subplots(figsize=(9, 4.8))
tRs = sorted({r[0] for r in e2_rows})
xs = np.arange(len(grans)); wbar = 0.8 / (2 * len(tRs))
cols = {3.0: C["hbf3"], 20.0: C["hbf20"], 50.0: C["hbf50"]}
i = 0
for tR in tRs:
    for mocs, hatch in [(HBF_MOCS, ""), (None, "////")]:
        ys = [next(r.achieved_bw * HBF_CHANNELS / 1e12 for t, g, m, _, r in e2_rows if t == tR and g == gg and m == mocs) for gg in grans]
        ax.bar(xs + (i - len(tRs)) * wbar + wbar / 2, ys, wbar * 0.92, color=cols[tR], hatch=hatch,
               edgecolor="#fcfcfb", label=f"tR={tR:g}µs, MOCS {'16,384/ch' if mocs else 'off'}")
        i += 1
ax.axhline(3.072, color="#c3c2b7", ls="--", lw=1); ax.text(xs[0] - 0.45, 3.1, "grade 3 유효 3.07 TB/s", fontsize=8, color="#52514e")
ax.set_xticks(xs); ax.set_xticklabels([f"{g} B" for g in grans]); ax.set_xlabel("요청 입도")
ax.set_ylabel("달성 대역폭 (TB/s)"); ax.set_title("E2. 요청 입도 × MOCS — 64 B(L2 line miss 방식)로는 HBF를 채울 수 없음")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "e2_granularity.png"), dpi=150); plt.close(fig)

# ============================================================================
# E3: NAND array parallelism (sense units per channel) — buffer can't fix this
# ============================================================================
print("E3: array parallelism")
table = ["| tR | sense unit/채널 (N) | array 상한 (해석식) | 달성 BW (cube, 버퍼 충분) | 95% 도달 버퍼 (총) |",
         "|---|---|---|---|---|"]
e3 = {}
for tR in [3.0, 20.0]:
    warm = max(150 * us, 4 * tR * us); meas = 300 * us
    pts = []
    for N in [16, 64, 256, 1024, 4096]:
        cfg = hbf_channel(tR, N)
        arr = N * 4 * KiB / (tR * us)
        cap = min(arr, HBF_CH_EFF_BW)
        lat = device_latency(cfg, 4 * KiB)
        base = little(cap, lat)
        wins = sorted({int(max(4 * KiB, base * f) // (4 * KiB)) * 4 * KiB for f in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]})
        rows = sweep(cfg, 4 * KiB, wins, warm, meas)
        k95 = knee(rows, cap, 0.95)
        best = max(r.achieved_bw for r in rows)
        pts.append((N, best * HBF_CHANNELS))
        table.append(f"| {tR:g} µs | {N} | {min(arr, HBF_CH_EFF_BW)*HBF_CHANNELS/1e9:.0f} GB/s | {best*HBF_CHANNELS/1e9:.0f} GB/s | {fmt_mb(None if k95 is None else k95*HBF_CHANNELS)} |")
    e3[tR] = pts
results_md.append(("E3. NAND array 병렬도(채널당 sense unit 수 N) — 버퍼로 해결되지 않는 상한", "\n".join(table)))
fig, ax = plt.subplots(figsize=(8, 4.5))
for tR, pts in e3.items():
    ax.plot([p[0] for p in pts], [p[1] / 1e12 for p in pts], marker="o", ms=5, color=cols[tR], label=f"tR={tR:g} µs")
ax.axhline(3.072, color="#c3c2b7", ls="--", lw=1); ax.text(16, 3.12, "grade 3 유효 3.07 TB/s", fontsize=8, color="#52514e")
ax.axvline(16, color="#c3c2b7", ls=":", lw=1); ax.text(17, 0.3, "OCP 기준 구성 N=16", fontsize=8, color="#52514e", rotation=90)
ax.set_xscale("log", base=2); ax.set_xlabel("채널당 동시 sense unit 수 N (die × bank/plane)"); ax.set_ylabel("달성 대역폭 (TB/s)")
ax.set_title("E3. 버퍼가 충분해도 array 병렬도 N이 모자라면 BW = N×4KiB/tR 에서 멈춤")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "e3_array_parallelism.png"), dpi=150); plt.close(fig)

# ============================================================================
# E4: tR jitter (bank-to-bank / read-retry variance) raises the buffer need
# ============================================================================
print("E4: jitter")
table = ["| tR | jitter | 95% 도달 버퍼 (총) | 99% 도달 버퍼 (총) | p99 latency (버퍼=Little×1.0) |", "|---|---|---|---|---|"]
for tR in [20.0]:
    warm = 4 * tR * us; meas = 300 * us
    for jit in [0.0, 0.3, 0.6]:
        cfg = hbf_channel(tR, NONLIMITING_BANKS, jitter=jit)
        base = little(HBF_CH_EFF_BW, device_latency(cfg, 4 * KiB))
        wins = sorted({int(base * f // (4 * KiB)) * 4 * KiB for f in [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]})
        rows = sweep(cfg, 4 * KiB, wins, warm, meas)
        k95 = knee(rows, HBF_CH_EFF_BW, 0.95); k99 = knee(rows, HBF_CH_EFF_BW, 0.99)
        r1 = min(rows, key=lambda r: abs(r.window_bytes - base))
        table.append(f"| {tR:g} µs | ±{jit*100:.0f}% | {fmt_mb(None if k95 is None else k95*HBF_CHANNELS)} | {fmt_mb(None if k99 is None else k99*HBF_CHANNELS)} | {r1.p99_latency/us:.1f} µs |")
results_md.append(("E4. tR 편차(jitter)가 필요 버퍼에 미치는 영향 (tR 20 µs)", "\n".join(table)))

# ============================================================================
# E5: analytic scenario table — what the GPU must hold, per attachment scenario
# ============================================================================
table = ["| 시나리오 | 장치 유효 BW | latency 가정 | 필요 in-flight 바이트 (BW×lat) | 4 KiB 요청 수 | 64 B 요청 수 | H100 L2 50 MB 대비 |",
         "|---|---|---|---|---|---|---|"]
scen = [
    ("6×HBM3e 전부 (4.8 TB/s)", 4.8e12, 0.55 * us),
    ("6×HBM3e, 부하 시 (latency 1 µs)", 4.8e12, 1.0 * us),
    ("HBF grade 3 직결, tR 3 µs", 3.072e12, 3.3 * us),
    ("HBF grade 3 직결, tR 20 µs", 3.072e12, 20.3 * us),
    ("HBF grade 3 직결, tR 50 µs", 3.072e12, 50.3 * us),
    ("통로 반분: HBF grade 2 (1.536 TB/s), tR 20 µs", 1.536e12, 20.3 * us),
    ("통로 반분: HBF grade 2 (1.536 TB/s), tR 3 µs", 1.536e12, 3.3 * us),
    ("HBF grade 1 (0.384 TB/s), tR 20 µs", 0.384e12, 20.3 * us),
]
for name, bw, lat in scen:
    B = bw * lat
    table.append(f"| {name} | {bw/1e12:.3f} TB/s | {lat/us:.2f} µs | {B/1e6:.1f} MB | {B/4096:,.0f} | {B/64:,.0f} | {100*B/50e6:.0f}% |")
results_md.append(("E5. 시나리오별 Little's law 요약 (해석식)", "\n".join(table)))

with open(os.path.join(OUT, "tables.md"), "w") as f:
    for title, body in results_md:
        f.write(f"### {title}\n\n{body}\n\n")
print("\nwritten", OUT)
