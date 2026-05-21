#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
DEVICE = "Jetson"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

ROOT_DIR   = BASE_DIR.parent
DEVICE_DIR = ROOT_DIR / "logs" / DEVICE
PLOTS_DIR  = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

POWER_CSV  = DEVICE_DIR / "history_ReTrain.csv"
BLOCKS_CSV = DEVICE_DIR / "waidplus_blocks_times.csv"
OUTPUT_PDF = PLOTS_DIR  / "power_timeseries.pdf"
OUTPUT_PNG = PLOTS_DIR  / "power_timeseries.png"

# ── Configuración ──────────────────────────────────────────────────────────
BLOCKS_TZ           = "Europe/Madrid"
ROLL_WINDOW         = 60
DOWNSAMPLE          = 5
BLOCK_MARKERS_EVERY = 10

COLOR_RAW    = "#01696f"
COLOR_SMOOTH = "#01696f"
COLOR_MARKER = "#da7101"

# ── Carga de datos ─────────────────────────────────────────────────────────
df_power = pd.read_csv(POWER_CSV)
df_power["timestamp"] = pd.to_datetime(df_power["last_changed"], utc=True)
df_power["power_W"]   = pd.to_numeric(df_power["state"], errors="coerce")
df_power = (df_power
            .dropna(subset=["power_W"])
            .sort_values("timestamp")
            .reset_index(drop=True))

df_blocks = pd.read_csv(BLOCKS_CSV)
run_id = df_blocks.iloc[0]["RunId"]

# ── Parseo de bloques ──────────────────────────────────────────────────────
blocks = []
n_blocks = (len(df_blocks.columns) - 1) // 2
for i in range(1, n_blocks + 1):
    ini_col, fin_col = f"B{i}_Ini", f"B{i}_Fin"
    if ini_col not in df_blocks.columns:
        break
    t_ini = (pd.to_datetime(df_blocks.iloc[0][ini_col])
               .tz_localize(BLOCKS_TZ).tz_convert("UTC"))
    t_fin = (pd.to_datetime(df_blocks.iloc[0][fin_col])
               .tz_localize(BLOCKS_TZ).tz_convert("UTC"))
    blocks.append({"block": i, "t_ini": t_ini, "t_fin": t_fin})

blocks_df = pd.DataFrame(blocks)

# ── Tiempo relativo en horas desde el primer bloque ────────────────────────
t0 = blocks_df["t_ini"].iloc[0]

df_power["t_h"] = (df_power["timestamp"] - t0).dt.total_seconds() / 3600.0
blocks_df["t_h_ini"] = (blocks_df["t_ini"] - t0).dt.total_seconds() / 3600.0
blocks_df["t_h_fin"] = (blocks_df["t_fin"] - t0).dt.total_seconds() / 3600.0

# ── Media móvil ────────────────────────────────────────────────────────────
df_power["power_smooth"] = (df_power["power_W"]
                             .rolling(window=ROLL_WINDOW, center=True, min_periods=1)
                             .mean())

# ── Downsample + recorte desde hora 0 ─────────────────────────────────────
df_plot = df_power.iloc[::DOWNSAMPLE].copy()
df_plot = df_plot[df_plot["t_h"] >= 0].copy()
t_max   = df_plot["t_h"].max()

# ── Figura ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Señal raw
ax.plot(df_plot["t_h"], df_plot["power_W"],
        color=COLOR_RAW, linewidth=0.6, alpha=0.25, label="Potencia raw (W)")

# Media móvil
ax.plot(df_plot["t_h"], df_plot["power_smooth"],
        color=COLOR_SMOOTH, linewidth=2.0, alpha=0.95,
        label=f"Media móvil ({ROLL_WINDOW} muestras)")

# ── Eje X: empieza en 0, tick cada hora ───────────────────────────────────
ax.set_xlim(left=0, right=t_max)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x)}h" if x == int(x) else f"{x:.1f}h"))

# Líneas verticales en inicio de bloques clave
key_blocks = blocks_df[blocks_df["block"] % BLOCK_MARKERS_EVERY == 0]
for _, row in key_blocks.iterrows():
    ax.axvline(x=row["t_h_ini"], color=COLOR_MARKER, linewidth=0.9,
               linestyle="--", alpha=0.65)
    ax.text(row["t_h_ini"], ax.get_ylim()[1] * 0.98,
            f'B{int(row["block"])}',
            fontsize=10, color=COLOR_MARKER, ha="center", va="top",
            rotation=90, alpha=0.8)

# Etiquetas y título
ax.set_xlabel("Time (h)", fontsize=14)
ax.set_ylabel("Power(W)", fontsize=14)
#ax.set_title(f"Potencia instantánea — {run_id}", fontsize=14, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=0, labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.grid(axis="x", linestyle=":",  linewidth=0.3, alpha=0.4, color="gray")

# Leyenda
legend_handles = [
    plt.Line2D([0], [0], color=COLOR_RAW,    lw=1,  alpha=0.4,label="Raw power (W)"),
    plt.Line2D([0], [0], color=COLOR_SMOOTH, lw=2.0,label=f"Moving average ({ROLL_WINDOW} samples)"),
    plt.Line2D([0], [0], color=COLOR_MARKER, lw=1, linestyle="--", alpha=0.7,label=f"Block onset (every {BLOCK_MARKERS_EVERY} blocks)"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="lower right",framealpha=0.9, edgecolor="#cccccc")

# Estadísticas al pie
p_mean = df_power["power_W"].mean()
p_std  = df_power["power_W"].std()
p_min  = df_power["power_W"].min()
p_max  = df_power["power_W"].max()
#stats_txt = (f"μ = {p_mean:.1f} W  |  σ = {p_std:.1f} W  | min = {p_min:.1f} W  |  max = {p_max:.1f} W")
#fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=9, color="#555555", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close()

print(f"PDF : {OUTPUT_PDF}")
print(f"PNG : {OUTPUT_PNG}")
print(f"Run ID    : {run_id}")
print(f"Muestras  : {len(df_power)}")
print(f"Bloques   : {len(blocks_df)}")
print(f"Duración  : {t_max:.2f} h")
print(f"μ         : {p_mean:.2f} W")
print(f"σ         : {p_std:.2f} W")
print(f"Rango     : [{p_min:.1f}, {p_max:.1f}] W")
