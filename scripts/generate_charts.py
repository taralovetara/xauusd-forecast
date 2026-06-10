#!/usr/bin/env python3
"""
Generate performance charts for the XAUUSD H1 Forecast project.

Creates 4 charts:
  1. Confidence-Based Filtering Results (bar + line)
  2. Model Comparison (grouped bar)
  3. XAUUSD Price with Indicators (multi-subplot)
  4. 3-Concept Framework Architecture (text infographic)
"""

import json
import os
import sys

import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHART_DIR = os.path.join(BASE_DIR, 'analysis', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(MODEL_DIR, 'training_results_v2.json')
H1_CSV_PATH  = os.path.join(DATA_DIR, 'XAUUSD_H1.csv')

# ── Load data ──────────────────────────────────────────────────────────
with open(RESULTS_PATH, 'r') as f:
    results = json.load(f)

base_models  = results['results']['base_models']
conf_filter  = results['results']['confidence_filtering']
overall_acc  = results['results']['overall_accuracy']
overall_auc  = results['results']['overall_auc']

# ── Colour palette ─────────────────────────────────────────────────────
BLUE      = '#4285F4'
ORANGE    = '#FF8C00'
GREEN     = '#34A853'
RED       = '#EA4335'
PURPLE    = '#9C27B0'
TEAL      = '#009688'
GREY      = '#9E9E9E'
BG_COLOR  = '#FAFAFA'


# =====================================================================
# CHART 1 — Confidence-Based Filtering Results
# =====================================================================
def chart_confidence_filtering():
    """Bar chart (accuracy) + line (coverage) with dual Y-axes."""

    thresholds = ['≥0.60', '≥0.65', '≥0.70', '≥0.80', 'All']
    acc_values = [
        conf_filter['0.6']['accuracy']  * 100,
        conf_filter['0.65']['accuracy'] * 100,
        conf_filter['0.7']['accuracy']  * 100,
        conf_filter['0.8']['accuracy']  * 100,
        overall_acc * 100,
    ]
    cov_values = [
        conf_filter['0.6']['pct_samples'],
        conf_filter['0.65']['pct_samples'],
        conf_filter['0.7']['pct_samples'],
        conf_filter['0.8']['pct_samples'],
        100.0,
    ]

    x = np.arange(len(thresholds))
    width = 0.50

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax1.set_facecolor(BG_COLOR)

    # Accuracy bars
    bars = ax1.bar(x, acc_values, width, color=BLUE, alpha=0.85, edgecolor='white',
                   linewidth=1.2, zorder=3, label='Accuracy (%)')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, color=BLUE, fontweight='bold')
    ax1.set_ylim(50, 105)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds, fontsize=12, fontweight='bold')

    # Bar labels
    for bar, val in zip(bars, acc_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=BLUE)

    # Coverage line on second Y-axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, cov_values, color=ORANGE, marker='o', markersize=9,
                    linewidth=2.5, zorder=4, label='Coverage (%)')
    ax2.set_ylabel('Coverage (%)', fontsize=13, color=ORANGE, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis='y', labelcolor=ORANGE)

    # Line labels
    for xi, cv in zip(x, cov_values):
        ax2.text(xi, cv + 3, f'{cv:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=ORANGE)

    # Title & legend
    ax1.set_title('XAUUSD H1 Prediction: Accuracy vs Coverage by Confidence Threshold',
                  fontsize=14, fontweight='bold', pad=18)
    ax1.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')

    # Combined legend
    handles = [bars, line[0]]
    labels  = ['Accuracy (%)', 'Coverage (%)']
    ax1.legend(handles, labels, loc='lower left', fontsize=11,
               framealpha=0.9, edgecolor='grey')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    path = os.path.join(CHART_DIR, 'chart1_confidence_filtering.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Chart 1 saved: {path}')


# =====================================================================
# CHART 2 — Model Comparison (grouped bar)
# =====================================================================
def chart_model_comparison():
    """Grouped bar chart comparing base models + ensemble on Accuracy & AUC-ROC."""

    models   = ['XGBoost', 'LightGBM', 'CatBoost', 'Stacked\nEnsemble']
    acc_vals = [
        base_models['xgboost']['accuracy']  * 100,
        base_models['lightgbm']['accuracy'] * 100,
        base_models['catboost']['accuracy'] * 100,
        overall_acc * 100,
    ]
    auc_vals = [
        base_models['xgboost']['auc'],
        base_models['lightgbm']['auc'],
        base_models['catboost']['auc'],
        overall_auc,
    ]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars1 = ax.bar(x - width/2, acc_vals, width, color=BLUE, alpha=0.85,
                   edgecolor='white', linewidth=1.2, label='Accuracy (%)')
    bars2 = ax.bar(x + width/2, [v * 100 for v in auc_vals], width, color=ORANGE,
                   alpha=0.85, edgecolor='white', linewidth=1.2, label='AUC-ROC (×100)')

    # Labels on bars
    for bar, val in zip(bars1, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=BLUE)
    for bar, val in zip(bars2, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 100 / 100 + 0.4,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=ORANGE)

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Base Model vs Ensemble Performance', fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim(55, 78)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor='grey')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight ensemble
    ax.axvspan(2.6, 3.4, color=GREEN, alpha=0.08)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, 'chart2_model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Chart 2 saved: {path}')


# =====================================================================
# CHART 3 — XAUUSD Price with Indicators
# =====================================================================
def chart_price_indicators():
    """3-subplot chart: Price + SMA, RSI, MACD — last 500 bars."""

    df = pd.read_csv(H1_CSV_PATH, parse_dates=['time'])
    df = df.tail(500).copy().reset_index(drop=True)

    # Compute indicators
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()

    # RSI 14
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line']   = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']

    # Time axis
    t = df['time']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG_COLOR,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # ── Subplot 1: Price + SMA ──
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(t, df['close'], color='#1A1A2E', linewidth=1.3, label='Close', zorder=3)
    ax1.plot(t, df['sma20'], color=BLUE, linewidth=1.0, alpha=0.8, label='SMA 20')
    ax1.plot(t, df['sma50'], color=ORANGE, linewidth=1.0, alpha=0.8, label='SMA 50')
    ax1.fill_between(t, df['sma20'], df['sma50'],
                     where=df['sma20'] >= df['sma50'],
                     color=GREEN, alpha=0.08, interpolate=True)
    ax1.fill_between(t, df['sma20'], df['sma50'],
                     where=df['sma20'] < df['sma50'],
                     color=RED, alpha=0.08, interpolate=True)
    ax1.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
    ax1.set_title('XAUUSD H1 — Price + Indicators (Last 500 Bars)',
                  fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelbottom=False)

    # ── Subplot 2: RSI ──
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.plot(t, df['rsi14'], color=PURPLE, linewidth=1.0, label='RSI(14)')
    ax2.axhline(70, color=RED, linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(30, color=GREEN, linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.fill_between(t, 70, df['rsi14'],
                     where=df['rsi14'] >= 70, color=RED, alpha=0.15, interpolate=True)
    ax2.fill_between(t, 30, df['rsi14'],
                     where=df['rsi14'] <= 30, color=GREEN, alpha=0.15, interpolate=True)
    ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(10, 90)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelbottom=False)

    # ── Subplot 3: MACD ──
    ax3 = axes[2]
    ax3.set_facecolor(BG_COLOR)
    ax3.plot(t, df['macd_line'],   color=TEAL,   linewidth=1.0, label='MACD Line')
    ax3.plot(t, df['macd_signal'], color=ORANGE, linewidth=1.0, label='Signal')
    colours = [GREEN if v >= 0 else RED for v in df['macd_hist']]
    ax3.bar(t, df['macd_hist'], color=colours, alpha=0.6, width=0.0008, label='Histogram')
    ax3.axhline(0, color='grey', linewidth=0.5)
    ax3.set_ylabel('MACD', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.grid(alpha=0.3, linestyle='--')

    fig.tight_layout()
    path = os.path.join(CHART_DIR, 'chart3_price_indicators.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Chart 3 saved: {path}')


# =====================================================================
# CHART 4 — 3-Concept Framework Architecture (text infographic)
# =====================================================================
def chart_framework_architecture():
    """Text/infographic showing the 3-Concept Framework + Stacked Ensemble."""

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Title ──
    ax.text(7, 8.6, 'XAUUSD H1 3-Concept Framework + Stacked Ensemble',
            fontsize=18, fontweight='bold', ha='center', va='center', color='#1A1A2E')

    # ── Box helper ──
    def draw_box(x, y, w, h, title, items, title_color, bg_color, border_color):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.15',
            facecolor=bg_color, edgecolor=border_color, linewidth=2.0, alpha=0.92)
        ax.add_patch(rect)
        # Title
        ax.text(x + w/2, y + h - 0.30, title,
                fontsize=13, fontweight='bold', ha='center', va='center', color=title_color)
        # Items
        for i, item in enumerate(items):
            ax.text(x + w/2, y + h - 0.65 - i * 0.32, item,
                    fontsize=9.5, ha='center', va='center', color='#333333')

    # ── Concept boxes ──
    concept_data = [
        ('CONCEPT 1: TREND', [
            'Price vs SMA(20,50,200) / EMA(9,21,55)',
            'SMA/EMA slopes & crossovers',
            'ADX(14) trend strength',
            'Ichimoku cloud position',
            'H1 trend alignment',
        ], BLUE, '#E3F2FD', BLUE),

        ('CONCEPT 2: MOMENTUM', [
            'RSI(7,14,21) + divergence',
            'MACD histogram & crossover',
            'Stochastic %K/%D signals',
            'Rate of change (5,10,20,60)',
            'Volume ratio + MFI(14)',
        ], ORANGE, '#FFF3E0', ORANGE),

        ('CONCEPT 3: CYCLE', [
            'Bollinger Bands width & squeeze',
            'ATR(14) normalized + regime',
            'Fibonacci retracement position',
            'Session encoding (London Fix,',
            '  COMEX open) + hour encoding',
        ], GREEN, '#E8F5E9', GREEN),
    ]

    box_w = 4.0
    box_h = 2.8
    gap   = 0.5
    start_x = (14 - (3 * box_w + 2 * gap)) / 2
    box_y = 5.2

    for i, (title, items, tc, bg, bc) in enumerate(concept_data):
        bx = start_x + i * (box_w + gap)
        draw_box(bx, box_y, box_w, box_h, title, items, tc, bg, bc)

    # ── Arrow labels from concepts to features ──
    ax.annotate('', xy=(7, 4.9), xytext=(7, 5.15),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.text(7, 5.02, '33 features each = 99 base features', fontsize=9,
            ha='center', va='center', color='#666666', style='italic')

    # ── Advanced Features box ──
    adv_x = 4.0
    adv_y = 2.4
    adv_w = 6.0
    adv_h = 2.2
    draw_box(adv_x, adv_y, adv_w, adv_h,
             'ADVANCED FEATURES (+73)',
             [
                 'Lagged features (5 key indicators × 4 lags)',
                 'Interaction features (6 cross-indicator products)',
                 'Rolling stats + Candlestick patterns',
                 'Signal strength composites, regime categories',
             ],
             PURPLE, '#F3E5F5', PURPLE)

    # ── Total features annotation ──
    ax.text(7, 2.15, '99 base + 73 advanced = 172 total features  →  Top 80 selected by Mutual Information',
            fontsize=10, ha='center', va='center', color='#555555', fontweight='bold')

    # ── Stacked Ensemble box ──
    ens_x = 4.0
    ens_y = 0.3
    ens_w = 6.0
    ens_h = 1.6
    draw_box(ens_x, ens_y, ens_w, ens_h,
             'STACKED ENSEMBLE',
             [
                 'Level 0:  XGBoost  |  LightGBM  |  CatBoost',
                 'Level 1:  Logistic Regression meta-learner',
                 'Result:  62.83% accuracy  |  0.6898 AUC-ROC',
             ],
             '#D32F2F', '#FFEBEE', '#D32F2F')

    # ── Arrows ──
    ax.annotate('', xy=(7, 2.05), xytext=(7, 2.35),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.annotate('', xy=(7, 1.95), xytext=(7, 2.15),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.annotate('', xy=(7, 0.15), xytext=(7, 0.25),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))

    # ── Key insight badge ──
    badge = mpatches.FancyBboxPatch(
        (10.5, 0.3), 3.2, 1.6, boxstyle='round,pad=0.15',
        facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2.0, alpha=0.95)
    ax.add_patch(badge)
    ax.text(12.1, 1.55, 'KEY INSIGHT', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#E65100')
    ax.text(12.1, 1.10, 'Confidence ≥0.70:', fontsize=10,
            ha='center', va='center', color='#333333', fontweight='bold')
    ax.text(12.1, 0.75, '93.96% accuracy', fontsize=13,
            ha='center', va='center', color=GREEN, fontweight='bold')
    ax.text(12.1, 0.45, 'on 5.6% of data', fontsize=9,
            ha='center', va='center', color='#666666')

    fig.tight_layout()
    path = os.path.join(CHART_DIR, 'chart4_framework_architecture.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Chart 4 saved: {path}')


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print('Generating XAUUSD Forecast Charts...\n')
    chart_confidence_filtering()
    chart_model_comparison()
    chart_price_indicators()
    chart_framework_architecture()
    print('\nAll charts generated successfully!')
