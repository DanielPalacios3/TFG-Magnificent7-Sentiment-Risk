"""
Script: generar_diagnosticos.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿De qué va este script?
------------------------
Este script genera 5 figuras de diagnóstico estadístico que son bastante importantes
para la defensa del TFG. Básicamente, lo que hacen es demostrar con datos reales que
nuestras series temporales cumplen las propiedades que justifican usar modelos GARCH
y XGBoost: hay heterocedasticidad (la varianza no es constante), hay clustering de
volatilidad (los periodos volátiles tienden a agruparse), y hay que vigilar la
multicolinealidad entre las variables predictoras.

Las 5 figuras que genera y para qué sirve cada una:
----------------------------------------------------
- docs/figuras/diagnostico_homo_vs_hetero.png
    Una comparación visual lado a lado entre datos con varianza constante (simulados)
    y datos reales de TSLA. Es la típica figura didáctica para que el tribunal
    entienda de un vistazo qué es la heterocedasticidad.

- docs/figuras/diagnostico_heterocedasticidad.png
    Aquí ya nos ponemos serios con TSLA: retornos diarios, volatilidad rolling de 20 días
    y retornos al cuadrado. Se ve clarísimo que la volatilidad cambia a lo largo del tiempo.

- docs/figuras/diagnostico_test_arch_lm.png
    El test formal ARCH-LM de Engle (1982) para las 7 empresas. Si el test sale
    significativo (p < 0.05), tenemos evidencia estadística de heterocedasticidad
    condicional, que es exactamente lo que justifica usar GARCH.

- docs/figuras/diagnostico_clustering_volatilidad.png
    Compara la autocorrelación de retornos (que es casi cero, o sea impredecibles) con
    la de retornos al cuadrado (que es positiva y persistente). Esto demuestra que aunque
    no podemos predecir la dirección, sí podemos predecir la magnitud de los movimientos.

- docs/figuras/diagnostico_multicolinealidad.png
    Heatmap de correlaciones entre las variables de sentimiento + VIF (factor de inflación
    de varianza). Sirve para detectar si hay variables redundantes que meten ruido en
    los modelos, como por ejemplo el tono medio de GDELT y el porcentaje de noticias
    negativas, que están bastante correlacionados entre sí.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Configuración ─────────────────────────────────────────────────────────────
# Rutas, tickers y ajustes globales de matplotlib

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / 'docs' / 'figuras'
FIGURAS.mkdir(parents=True, exist_ok=True)

RUTA_DATASET = RAIZ / 'data' / 'clean' / 'dataset_integrado.csv'
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})


def guardar_figura(nombre):
    """Guarda la figura de matplotlib que esté activa en ese momento como PNG y la cierra
    para liberar memoria. Imprime un tick para que sepamos que fue bien."""
    ruta = FIGURAS / nombre
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {nombre}")


# ── Carga de datos ────────────────────────────────────────────────────────────
# Leemos el dataset integrado que tiene toda la info de mercado, noticias y Reddit juntas

print("Cargando dataset...")
df = pd.read_csv(RUTA_DATASET, parse_dates=['date'])
print(f"  {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"  Periodo: {df['date'].min().date()} → {df['date'].max().date()}")


# =========================================================================
# FIGURA 1: Homocedasticidad vs Heterocedasticidad — la comparación visual
# Esta es la figura "para dummies": a la izquierda datos con varianza
# constante (simulados), a la derecha datos reales de TSLA donde se ve
# clarísimo que la volatilidad va y viene. Perfecta para abrir la defensa.
# =========================================================================
print("\nGenerando figuras...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

np.random.seed(42)
n = 500

# Panel izquierdo: datos simulados con varianza constante (lo que NO tenemos)
ax1 = axes[0]
homo_data = np.random.normal(0, 1, n)
ax1.plot(homo_data, color='#3498db', linewidth=0.8, alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axhline(2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axhline(-2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.fill_between(range(n), -2, 2, alpha=0.05, color='red')
ax1.set_title('HOMOCEDASTICIDAD\n(varianza constante — NO es nuestro caso)',
              fontsize=12, fontweight='bold', color='#3498db')
ax1.set_ylabel('Valor')
ax1.set_xlabel('Tiempo')
ax1.set_ylim(-6, 6)
ax1.text(250, 4.5, 'Dispersión uniforme\nen todo el periodo', ha='center',
         fontsize=10, style='italic', color='#3498db')

# Panel derecho: datos reales de TSLA (lo que SÍ tenemos — varianza que cambia con el tiempo)
ax2 = axes[1]
dt = df[df['ticker'] == 'TSLA'].sort_values('date')
returns_tsla = dt['log_return'].dropna().values[-500:]
returns_norm = returns_tsla / np.std(returns_tsla)
ax2.plot(returns_norm, color='#e74c3c', linewidth=0.8, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title('HETEROCEDASTICIDAD\n(varianza cambiante — NUESTROS DATOS)',
              fontsize=12, fontweight='bold', color='#e74c3c')
ax2.set_ylabel('Retorno normalizado')
ax2.set_xlabel('Tiempo (últimos 500 días de TSLA)')
ax2.set_ylim(-6, 6)

for i in range(len(returns_norm)):
    if abs(returns_norm[i]) > 2.5:
        ax2.axvspan(max(0, i-10), min(len(returns_norm), i+10), alpha=0.08, color='red')

ax2.text(250, 4.5, 'Periodos tranquilos\ny periodos convulsos', ha='center',
         fontsize=10, style='italic', color='#e74c3c')

plt.suptitle('¿Qué es heterocedasticidad? — Comparación visual',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('diagnostico_homo_vs_hetero.png')


# =========================================================================
# FIGURA 2: Heterocedasticidad con datos reales de TSLA
# Tres paneles apilados que cuentan la misma historia desde ángulos distintos:
# los retornos diarios (con movimientos extremos en rojo), la volatilidad
# rolling de 20 días, y los retornos al cuadrado donde se ven los clusters.
# =========================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                         gridspec_kw={'height_ratios': [2, 1, 1]})

ticker = 'TSLA'
dt = df[df['ticker'] == ticker].sort_values('date')
returns = dt['log_return'].values
dates = dt['date'].values

# Panel superior: retornos diarios de TSLA (los que superan el 5% van en rojo)
ax = axes[0]
colors = ['#e74c3c' if abs(r) > 0.05 else '#3498db' for r in returns]
ax.bar(dates, returns, color=colors, width=1.5, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Retorno diario')
ax.set_title(f'HETEROCEDASTICIDAD en {ticker} — La volatilidad NO es constante',
             fontsize=14, fontweight='bold')
ax.text(0.02, 0.95, 'Rojo = movimiento > 5%', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', color='#e74c3c', fontweight='bold')

for label, start, end in [('COVID', '2020-02-20', '2020-04-15'),
                           ('Aranceles', '2025-03-15', '2025-04-02')]:
    mask = (dt['date'] >= start) & (dt['date'] <= end)
    if mask.any():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color='red')
        ax.text(pd.Timestamp(start), ax.get_ylim()[1]*0.85, label, fontsize=8, color='red')

# Panel medio: volatilidad realizada con ventana de 20 días — se ve cómo sube y baja
ax2 = axes[1]
ax2.fill_between(dates, dt['vol_realized_20d'].values, alpha=0.4, color='darkorange')
ax2.plot(dates, dt['vol_realized_20d'].values, color='darkorange', linewidth=1)
ax2.set_ylabel('Vol. 20d (anualizada)')
ax2.set_title('Volatilidad realizada 20 días — varía enormemente en el tiempo', fontsize=11)
mean_vol = dt['vol_realized_20d'].mean()
ax2.axhline(mean_vol, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.text(dates[-50], mean_vol + 0.05, f'media={mean_vol:.2f}', fontsize=8)

# Panel inferior: retornos al cuadrado — aquí se ve el clustering de volatilidad muy claro
ax3 = axes[2]
sq_ret = dt['log_return'].values**2
ax3.bar(dates, sq_ret, color='#9b59b6', width=1.5, alpha=0.7)
ax3.set_ylabel('Retorno²')
ax3.set_title('Retornos al cuadrado — los picos se agrupan (clustering de volatilidad)',
              fontsize=11)
ax3.set_xlabel('Fecha')

plt.tight_layout()
guardar_figura('diagnostico_heterocedasticidad.png')


# =========================================================================
# FIGURA 3: Test ARCH-LM de Engle (1982)
# Este es el test formal que nos dice si hay efectos ARCH en los retornos.
# Si el p-valor es menor que 0.05, rechazamos la hipótesis nula de varianza
# constante, lo que justifica formalmente usar modelos GARCH. Spoiler: todas
# las empresas salen súper significativas, como era de esperar.
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

lm_stats = {}
for ticker in TICKERS:
    dt = df[df['ticker'] == ticker].sort_values('date')
    returns = dt['log_return'].dropna().values
    stat, p, _, _ = het_arch(returns, nlags=5)
    lm_stats[ticker] = {'stat': stat, 'p': p}

tickers_sorted = sorted(lm_stats.keys(), key=lambda t: lm_stats[t]['stat'], reverse=True)
stats_vals = [lm_stats[t]['stat'] for t in tickers_sorted]
pvals = [lm_stats[t]['p'] for t in tickers_sorted]
colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.05 else '#95a5a6' for p in pvals]

ax.barh(range(len(tickers_sorted)), stats_vals, color=colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(tickers_sorted)))
ax.set_yticklabels(tickers_sorted, fontsize=12, fontweight='bold')
ax.set_xlabel('Estadístico ARCH-LM (mayor = más heterocedasticidad)', fontsize=11)
ax.set_title('Test ARCH-LM de Engle (1982) — ¿Varianza constante?',
             fontsize=14, fontweight='bold')

for i, (stat, p) in enumerate(zip(stats_vals, pvals)):
    label = f'p={p:.0e}' if p < 0.001 else f'p={p:.3f}'
    ax.text(stat + 5, i, label, va='center', fontsize=9, fontweight='bold')

ax.axvline(x=11.07, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.text(13, len(tickers_sorted)-0.5, 'Umbral\nrechazo\n(p=0.05)', fontsize=8, ha='left')

legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', label='Heterocedasticidad (p<0.001)'),
    mpatches.Patch(facecolor='#95a5a6', label='No significativo (p>0.05)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
guardar_figura('diagnostico_test_arch_lm.png')


# =========================================================================
# FIGURA 4: Autocorrelación de retornos vs retornos al cuadrado
# Esta es una de las figuras más reveladoras: arriba se ve que los retornos
# apenas tienen autocorrelación (no puedes predecir si mañana sube o baja),
# pero abajo los retornos al cuadrado sí que tienen autocorrelación fuerte
# (si hoy hubo un movimiento grande, mañana probablemente también lo habrá).
# Eso es exactamente lo que GARCH intenta modelar.
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for idx, ticker in enumerate(['AAPL', 'NVDA', 'TSLA']):
    dt = df[df['ticker'] == ticker].sort_values('date')
    returns = dt['log_return'].dropna()

    # Fila de arriba: autocorrelación de retornos — debería ser cercana a cero
    ax_top = axes[0, idx]
    acf_vals = [returns.autocorr(lag=k) for k in range(1, 21)]
    ax_top.bar(range(1, 21), acf_vals, color='#3498db', alpha=0.7)
    ax_top.axhline(1.96/np.sqrt(len(returns)), color='red', linestyle='--', linewidth=0.8)
    ax_top.axhline(-1.96/np.sqrt(len(returns)), color='red', linestyle='--', linewidth=0.8)
    ax_top.set_ylim(-0.15, 0.35)
    ax_top.set_title(f'{ticker} — Retornos', fontsize=12, fontweight='bold')
    if idx == 0:
        ax_top.set_ylabel('Autocorrelación')
    ax_top.text(10, 0.28, 'AC ≈ 0\n(impredecible)', ha='center', fontsize=9,
                color='#3498db', fontweight='bold')

    # Fila de abajo: autocorrelación de retornos al cuadrado — aquí sí que hay correlación
    ax_bot = axes[1, idx]
    returns2 = returns**2
    acf_vals2 = [returns2.autocorr(lag=k) for k in range(1, 21)]
    ax_bot.bar(range(1, 21), acf_vals2, color='#e74c3c', alpha=0.7)
    ax_bot.axhline(1.96/np.sqrt(len(returns2)), color='red', linestyle='--', linewidth=0.8)
    ax_bot.axhline(-1.96/np.sqrt(len(returns2)), color='red', linestyle='--', linewidth=0.8)
    ax_bot.set_ylim(-0.15, 0.35)
    ax_bot.set_title(f'{ticker} — Retornos²', fontsize=12, fontweight='bold')
    ax_bot.set_xlabel('Lag (días)')
    if idx == 0:
        ax_bot.set_ylabel('Autocorrelación')
    ax_bot.text(10, 0.28, 'AC > 0\n(PREDECIBLE)', ha='center', fontsize=9,
                color='#e74c3c', fontweight='bold')

fig.suptitle('Clustering de Volatilidad — Retornos impredecibles, pero su magnitud persiste',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('diagnostico_clustering_volatilidad.png')


# =========================================================================
# FIGURA 5: Multicolinealidad entre variables predictoras
# Dos paneles: a la izquierda un heatmap de correlaciones entre las variables
# de sentimiento (para ver de un vistazo cuáles están muy correlacionadas),
# y a la derecha un gráfico de barras con el VIF de cada variable. Si el VIF
# es mayor que 5, esa variable tiene multicolinealidad problemática y habría
# que plantearse quitarla o combinarla con otra.
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel izquierdo: heatmap de correlaciones entre las variables de sentimiento
sent_cols = ['news_tone_mean', 'news_tone_std', 'news_n_noticias', 'news_pct_negativo',
             'rddt_n_posts', 'rddt_score_mean', 'rddt_upvote_ratio_mean']
labels_cortas = ['Tono GDELT', 'Tono std', 'N. noticias', '% Negativo',
                 'Posts Reddit', 'Score Reddit', 'Upvote ratio']

corr = df[sent_cols].corr()
im = axes[0].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(labels_cortas)))
axes[0].set_yticks(range(len(labels_cortas)))
axes[0].set_xticklabels(labels_cortas, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(labels_cortas, fontsize=9)
axes[0].set_title('Correlación entre features de sentimiento', fontsize=12, fontweight='bold')

for i in range(len(sent_cols)):
    for j in range(len(sent_cols)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

# Resaltamos con un recuadro amarillo el par de variables que están peligrosamente correlacionadas
rect = plt.Rectangle((3-0.5, 0-0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=3)
axes[0].add_patch(rect)
rect2 = plt.Rectangle((0-0.5, 3-0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=3)
axes[0].add_patch(rect2)
axes[0].text(3, -0.8, 'ρ = -0.89\n⚠ Colineales', ha='center', fontsize=8,
             color='red', fontweight='bold')
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Panel derecho: VIF (factor de inflación de varianza) para cada variable predictora
features_vif = ['news_tone_mean', 'news_pct_negativo', 'rddt_n_posts',
                'rddt_upvote_ratio_mean', 'vol_realized_5d', 'vol_intraday']
labels_vif = ['Tono GDELT', '% Negativo GDELT', 'Posts Reddit',
              'Upvote ratio', 'Vol. 5d', 'Vol. intradía']

df_vif = df[features_vif].dropna()
df_vif_std = (df_vif - df_vif.mean()) / df_vif.std()

vifs = []
for i in range(len(features_vif)):
    vifs.append(variance_inflation_factor(df_vif_std.values, i))

colors_vif = ['#e74c3c' if v > 5 else '#f39c12' if v > 2.5 else '#2ecc71' for v in vifs]
axes[1].barh(range(len(features_vif)), vifs, color=colors_vif, edgecolor='white', height=0.6)
axes[1].set_yticks(range(len(features_vif)))
axes[1].set_yticklabels(labels_vif, fontsize=10)
axes[1].set_xlabel('VIF (Factor de Inflación de Varianza)', fontsize=11)
axes[1].set_title('Multicolinealidad — VIF por variable', fontsize=12, fontweight='bold')
axes[1].axvline(x=5, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axvline(x=2.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
axes[1].text(5.1, 5.5, 'Umbral alto\n(VIF>5)', fontsize=8, color='red')
axes[1].text(2.6, 5.5, 'Moderado\n(VIF>2.5)', fontsize=8, color='orange')

for i, v in enumerate(vifs):
    axes[1].text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=10, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', label='Sin problema (VIF<2.5)'),
    mpatches.Patch(facecolor='#e74c3c', label='Multicolinealidad alta (VIF>5)')
]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
axes[1].grid(True, axis='x', alpha=0.3)

plt.suptitle('Diagnóstico de Multicolinealidad entre Variables Predictoras',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('diagnostico_multicolinealidad.png')


# ── Resumen final ─────────────────────────────────────────────────────────────
# Mostramos qué figuras se generaron para confirmar que todo fue bien

print(f"""
{'='*60}
  FIGURAS DIAGNÓSTICAS GENERADAS
{'='*60}
  1. diagnostico_homo_vs_hetero.png
  2. diagnostico_heterocedasticidad.png
  3. diagnostico_test_arch_lm.png
  4. diagnostico_clustering_volatilidad.png
  5. diagnostico_multicolinealidad.png

  Directorio: {FIGURAS}
{'='*60}
""")
