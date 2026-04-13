"""
Script: clean_market_intraday.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Este script hace lo mismo que clean_market.py pero para los datos horarios (intradía).
La idea es limpiar las barras de una hora que descargamos con download_market_intraday.py
y generar las figuras comparativas de antes/después que van al documento Word.

La limpieza intradía tiene sus particularidades respecto a los datos diarios:
1. El índice es un datetime con zona horaria (UTC), no una fecha simple como en diario.
2. Los "huecos" entre las 16:00 de un día y las 9:30 del siguiente NO son errores: el
   mercado cierra por la noche y los fines de semana. Hay que tener cuidado de no
   confundirlos con datos faltantes.
3. Las ventanas rolling son más cortas (6h y 30h) en vez de 5 y 20 días.
4. Añadimos variables temporales (hora del día, día de la semana) porque hay patrones
   intradía muy conocidos, como el "efecto U" de volatilidad alta en apertura y cierre.
5. La primera barra de cada ticker no tiene retorno previo, así que se quita.

Archivos que genera
-------------------
- data/clean/market_intraday_clean.csv                  → el dataset limpio
- docs/figuras/intraday_nulos_antes.png                 → estado de nulos antes de limpiar
- docs/figuras/intraday_nulos_despues.png               → estado de nulos después
- docs/figuras/intraday_retornos_boxplot.png            → distribución de retornos horarios
- docs/figuras/intraday_volatilidad_serie.png           → cómo evoluciona la volatilidad
- docs/figuras/intraday_patron_horario.png              → el famoso efecto U de volatilidad
- docs/figuras/intraday_correlacion_volatilidad.png     → correlación entre empresas
- docs/figuras/intraday_estadisticas.png                → resumen visual de stats

Sobre las decisiones que tomamos
--------------------------------
- Los NaN en vol_realized_6h y vol_realized_30h se conservan tal cual. Son el mismo tema
  que en diario: la ventana rolling necesita datos previos y al principio no los tiene.
- La primera barra de cada ticker se elimina (NaN en log_return_1h). Sin barra anterior
  no hay retorno que calcular.
- Las brechas nocturnas y de fines de semana se dejan como están, obviamente.
- Los outliers no se tocan, por la misma razón que en diario.
- La zona horaria se convierte a hora del Este (America/New_York) para que los patrones
  intradía sean interpretables (el mercado abre a las 9:30 ET, no UTC).

Para ejecutar
-------------
    source venv/bin/activate
    python scripts/clean_market_intraday.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuracion general ────────────────────────────────────────────────────
# Rutas, tickers, colores corporativos y tamaño de ventanas rolling.
# Las ventanas deben coincidir con las que usamos en download_market_intraday.py.

RUTA_RAW   = os.path.join("data", "raw", "market_intraday", "ALL_market_intraday_2024_2026.csv")
RUTA_CLEAN = os.path.join("data", "clean", "market_intraday_clean.csv")
DIR_FIGS   = os.path.join("docs", "figuras")

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
COLORES = {
    "AAPL": "#555555", "MSFT": "#0078D4", "GOOGL": "#EA4335",
    "AMZN": "#FF9900", "NVDA": "#76B900", "META": "#1877F2", "TSLA": "#CC0000",
}

# Tamaño de las ventanas rolling en horas (tienen que ser iguales a las de la descarga)
VENTANA_CORTA = 6
VENTANA_LARGA = 30

os.makedirs(DIR_FIGS, exist_ok=True)


# ── Funciones auxiliares ──────────────────────────────────────────────────────
# Las mismas utilidades que en clean_market.py: separador bonito para la
# terminal y función para guardar figuras en la carpeta docs/figuras/.

def separador(titulo: str):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print('='*65)


def guardar_figura(nombre: str):
    ruta = os.path.join(DIR_FIGS, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figura guardada: {ruta}")


# ── 1. Carga del CSV en crudo ─────────────────────────────────────────────────
# Leemos los datos intradía y nos aseguramos de que el datetime tenga zona
# horaria UTC. Luego lo ponemos como índice y lo ordenamos cronológicamente.

separador("1. CARGA DE DATOS")

df_raw = pd.read_csv(RUTA_RAW)
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True)

df_raw = df_raw.set_index("datetime").sort_index()

print(f"  Filas cargadas : {len(df_raw):,}")
print(f"  Columnas       : {df_raw.columns.tolist()}")
print(f"  Tickers        : {sorted(df_raw['ticker'].unique())}")
print(f"  Rango fechas   : {df_raw.index.min()} → {df_raw.index.max()}")
print(f"  Barras por ticker:\n{df_raw.groupby('ticker').size().to_string()}")


# ── 2. Revisión inicial (ANTES de tocar nada) ────────────────────────────────
# Repasamos nulos, duplicados, coherencia OHLCV y distribución horaria para
# asegurarnos de que no hay barras fuera del horario de mercado.

separador("2. REVISIÓN INICIAL — ANTES DE LIMPIEZA")

# 2a. Contamos cuántos nulos hay por columna
nulos_antes = df_raw.isnull().sum()
print(f"\n  Valores nulos por columna:")
print(nulos_antes[nulos_antes > 0].to_string())

# 2b. Buscamos duplicados por combinación datetime + ticker
dupl = df_raw.reset_index().duplicated(["datetime", "ticker"]).sum()
print(f"\n  Duplicados (datetime+ticker): {dupl}")

# 2c. Verificamos que los precios OHLCV sean coherentes (High >= Low, etc.)
inc_hl = (df_raw["high"] < df_raw["low"]).sum()
inc_ch = (df_raw["close"] > df_raw["high"]).sum()
inc_cl = (df_raw["close"] < df_raw["low"]).sum()
print(f"\n  Coherencia OHLCV:")
print(f"    High < Low   : {inc_hl}")
print(f"    Close > High : {inc_ch}")
print(f"    Close < Low  : {inc_cl}")

# 2d. Miramos qué horas aparecen en los datos para confirmar que no se han
# colado barras de fuera del horario de mercado (antes de las 9:30 o después
# de las 16:00 ET).
horas_presentes = df_raw.index.hour.value_counts().sort_index()
print(f"\n  Horas presentes en el dataset (UTC):\n{horas_presentes.to_string()}")

# Figura ANTES: mostramos qué porcentaje de cada columna tiene datos completos
nulos_conteo = df_raw.isnull().sum()
completitud  = (1 - df_raw.isnull().mean()) * 100

fig, ax = plt.subplots(figsize=(9, 6))
colores_bar = ["#e53935" if v < 100 else "#43a047" for v in completitud.values]
ax.barh(completitud.index, completitud.values, color=colores_bar, edgecolor="white")
ax.set_xlim(0, 108)
ax.set_xlabel("% de barras con valor (completitud)")
ax.set_title("Completitud de datos intradía — ANTES de limpieza\n"
             "(rojo = columnas con NaN; verde = 100% completo)", fontsize=11)
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)

for i, (col, pct) in enumerate(completitud.items()):
    n_nan = int(nulos_conteo[col])
    etiqueta = f"{pct:.2f}%  ({n_nan} NaN)" if n_nan > 0 else "100%"
    ax.text(pct + 0.3, i, etiqueta, va="center", fontsize=7.5,
            color="#c62828" if n_nan > 0 else "#2e7d32")

plt.tight_layout()
guardar_figura("intraday_nulos_antes.png")


# ── 3. Limpieza ───────────────────────────────────────────────────────────────
# Aquí aplicamos las transformaciones. Al igual que en diario, hay poco que
# limpiar, pero hacemos todas las verificaciones por responsabilidad.

separador("3. LIMPIEZA Y TRANSFORMACIONES")

df = df_raw.copy()

# 3a. Quitamos duplicados si los hay
n_antes = len(df)
df = df.reset_index().drop_duplicates(["datetime", "ticker"]).set_index("datetime")
print(f"  Duplicados eliminados: {n_antes - len(df)}")

# 3b. Quitamos la primera barra de cada ticker porque no tiene retorno
# calculable (no hay barra anterior con la que comparar el precio)
n_antes = len(df)
primera_dt_por_ticker = df.groupby("ticker").apply(lambda g: g.index.min())
mask_primera = df.reset_index().apply(
    lambda row: row["datetime"] == primera_dt_por_ticker[row["ticker"]], axis=1
).values
df = df[~mask_primera].copy()
print(f"  Primeras barras eliminadas (sin log_return_1h): {n_antes - len(df)} "
      f"(1 por cada uno de los {len(TICKERS)} tickers — esperado)")

# 3c. Los NaN de las ventanas rolling se documentan pero no se eliminan.
# Mismo razonamiento que en diario: la ventana necesita datos previos.
nan_corta = df[f"vol_realized_{VENTANA_CORTA}h"].isna().sum()
nan_larga = df[f"vol_realized_{VENTANA_LARGA}h"].isna().sum()
print(f"\n  NaN vol_realized_{VENTANA_CORTA}h  : {nan_corta}  "
      f"(esperados: {len(TICKERS) * (VENTANA_CORTA - 1)} = {len(TICKERS)} tickers × {VENTANA_CORTA - 1} barras)")
print(f"  NaN vol_realized_{VENTANA_LARGA}h : {nan_larga} "
      f"(esperados: {len(TICKERS) * (VENTANA_LARGA - 1)} = {len(TICKERS)} tickers × {VENTANA_LARGA - 1} barras)")
print(f"  → DECISIÓN: se conservan. Son artefactos de ventana rolling, no errores.")

# 3d. Convertimos el índice a hora del Este (America/New_York). Esto es
# importante porque el mercado abre a las 9:30 ET, no UTC. Python gestiona
# automáticamente el cambio horario verano/invierno (EDT vs EST).
df.index = df.index.tz_convert("America/New_York")
df.index.name = "datetime_et"
print(f"\n  Índice convertido a hora ET (America/New_York)")

# 3e. Ajustamos tipos de datos y añadimos columnas de hora y día de la semana
# para poder analizar patrones intradía (por ejemplo, si la volatilidad es
# mayor a primera hora o al cierre)
df["ticker"]  = df["ticker"].astype("category")
df["volume"]  = df["volume"].astype("int64")
df["hora"]    = df.index.hour
df["dia_semana"] = df.index.dayofweek

# 3f. Ordenamos por ticker y hora para que quede todo secuencial
df = df.sort_values(["ticker", "datetime_et"])

print(f"\n  Shape final: {df.shape}")
print(f"  Rango: {df.index.min()} → {df.index.max()}")


# ── 4. Figuras DESPUÉS de la limpieza ────────────────────────────────────────
# Generamos las figuras con los datos ya limpios para comparar con el antes.

separador("4. FIGURAS DESPUÉS DE LIMPIEZA")

# 4a. Completitud DESPUÉS: misma estructura que antes para comparar fácilmente
nulos_d     = df.isnull().sum()
completitud_d = (1 - df.isnull().mean()) * 100

fig, ax = plt.subplots(figsize=(9, 6))
colores_bar_d = ["#e53935" if v < 100 else "#43a047" for v in completitud_d.values]
ax.barh(completitud_d.index, completitud_d.values, color=colores_bar_d, edgecolor="white")
ax.set_xlim(0, 108)
ax.set_xlabel("% de barras con valor (completitud)")
ax.set_title("Completitud de datos intradía — DESPUÉS de limpieza\n"
             "(los NaN restantes son artefactos inevitables de ventana rolling)", fontsize=11)
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)

for i, (col, pct) in enumerate(completitud_d.items()):
    n_nan = int(nulos_d[col])
    etiqueta = f"{pct:.2f}%  ({n_nan} NaN — rolling)" if n_nan > 0 else "100%  ✓"
    ax.text(pct + 0.3, i, etiqueta, va="center", fontsize=7.5,
            color="#c62828" if n_nan > 0 else "#2e7d32")

plt.tight_layout()
guardar_figura("intraday_nulos_despues.png")

# 4b. Boxplot de retornos logarítmicos a escala horaria. Los outliers aquí
# son movimientos bruscos dentro de una sola hora, lo cual es bastante extremo.
fig, ax = plt.subplots(figsize=(12, 5))
data_box = [df[df["ticker"] == t]["log_return_1h"].dropna().values for t in TICKERS]
bp = ax.boxplot(data_box, labels=TICKERS, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch, ticker in zip(bp["boxes"], TICKERS):
    patch.set_facecolor(COLORES[ticker])
    patch.set_alpha(0.7)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Distribución de retornos logarítmicos horarios por empresa (2024–2026)\n"
             "Los puntos extremos representan episodios de alta volatilidad intradía",
             fontsize=11)
ax.set_ylabel("Retorno logarítmico (barra 1h)")
ax.set_xlabel("Empresa")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.tight_layout()
guardar_figura("intraday_retornos_boxplot.png")

# 4c. Series temporales de volatilidad realizada a 30h por empresa.
# 30 barras de 1 hora equivalen a unos 5 días de trading, así que es
# comparable a la volatilidad de 20d diaria pero a escala intradía.
fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=False)
axes_flat = axes.flatten()

for i, ticker in enumerate(TICKERS):
    ax = axes_flat[i]
    sub = df[df["ticker"] == ticker].sort_index()

    ax.fill_between(sub.index, sub[f"vol_realized_{VENTANA_LARGA}h"],
                    color=COLORES[ticker], alpha=0.25)
    ax.plot(sub.index, sub[f"vol_realized_{VENTANA_LARGA}h"],
            color=COLORES[ticker], linewidth=0.8)

    ax.set_title(ticker, fontsize=12, fontweight="bold", color=COLORES[ticker])
    ax.set_ylabel(f"Vol. {VENTANA_LARGA}h", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

axes_flat[-1].set_visible(False)
fig.suptitle(f"Volatilidad realizada a {VENTANA_LARGA}h (≈5 días) — Magnificent 7 (2024–2026, horario)\n"
             "Anualizada con factor √(252×6.5) ≈ 40.47", fontsize=12, y=1.01)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
guardar_figura("intraday_volatilidad_serie.png")

# 4d. Patrón horario de volatilidad: la media por hora del día (de 9:30 a 16:00 ET).
# Este gráfico captura el famoso "efecto U" que describe la microestructura de mercado:
# la volatilidad es alta en la apertura (mucha información acumulada overnight), baja
# al mediodía (la gente se va a comer) y vuelve a subir al cierre (rebalanceos).
vol_por_hora = (
    df.groupby(["ticker", "hora"])[f"vol_realized_{VENTANA_LARGA}h"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
for ticker in TICKERS:
    sub = vol_por_hora[vol_por_hora["ticker"] == ticker]
    ax.plot(sub["hora"], sub[f"vol_realized_{VENTANA_LARGA}h"],
            marker="o", markersize=4,
            color=COLORES[ticker], label=ticker, linewidth=1.5)

ax.set_xlabel("Hora del día (ET)")
ax.set_ylabel(f"Volatilidad realizada media (ventana {VENTANA_LARGA}h)")
ax.set_title("Patrón horario de volatilidad realizada — Magnificent 7\n"
             "Efecto U: mayor volatilidad en apertura (9h–10h ET) y cierre (15h–16h ET)",
             fontsize=11)
ax.legend(ncol=4, fontsize=8, loc="upper center")
ax.grid(axis="both", linestyle=":", alpha=0.4)
# Sombreamos las zonas de apertura y cierre para que se vea claramente
ax.axvspan(9, 10.5, alpha=0.07, color="green", label="Apertura")
ax.axvspan(14.5, 16, alpha=0.07, color="red", label="Cierre")
plt.tight_layout()
guardar_figura("intraday_patron_horario.png")

# 4e. Heatmap de correlación entre volatilidades a 30h. Si es alta,
# las empresas se mueven juntas a nivel intradía (factor de mercado común).
fig, ax = plt.subplots(figsize=(8, 6))
vol_pivot = df.pivot_table(index="datetime_et", columns="ticker",
                           values=f"vol_realized_{VENTANA_LARGA}h")
corr = vol_pivot.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title(f"Correlación de volatilidad realizada {VENTANA_LARGA}h entre empresas\n"
             "Alta correlación indica movimiento conjunto (factor mercado)", fontsize=10)
plt.tight_layout()
guardar_figura("intraday_correlacion_volatilidad.png")


# ── 5. Estadísticas descriptivas ──────────────────────────────────────────────
# Estadísticas a escala horaria: retornos, volatilidad, rango y retorno
# absoluto. Ojo que los números son mucho más pequeños que en diario porque
# cada barra solo cubre una hora de trading.

separador("5. ESTADÍSTICAS DESCRIPTIVAS POR EMPRESA")

cols_stats = [
    "log_return_1h",
    f"vol_realized_{VENTANA_LARGA}h",
    "range_proxy_1h",
    "abs_return",
]
stats = df.groupby("ticker")[cols_stats].agg(["mean", "std", "min", "max"]).round(5)
print(stats.to_string())

print("\n  Resumen ejecutivo (escala horaria):")
print(f"  {'Ticker':<6} {'Ret.medio/h':>12} {'Vol.{v}h media':>16} {'Peor barra':>11} {'Mejor barra':>12}".format(v=VENTANA_LARGA))
print(f"  {'-'*65}")
for ticker in TICKERS:
    sub = df[df["ticker"] == ticker]
    ret_medio = sub["log_return_1h"].mean()
    vol_media = sub[f"vol_realized_{VENTANA_LARGA}h"].mean()
    peor      = sub["log_return_1h"].min()
    mejor     = sub["log_return_1h"].max()
    print(f"  {ticker:<6} {ret_medio:>+11.6f}  {vol_media:>15.4f}  {peor:>+10.4f}  {mejor:>+11.4f}")

# Figura resumen: volatilidad media y retorno absoluto medio por empresa
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

vol_media = df.groupby("ticker")[f"vol_realized_{VENTANA_LARGA}h"].mean().sort_values(ascending=False)
axes[0].bar(vol_media.index, vol_media.values,
            color=[COLORES[t] for t in vol_media.index], alpha=0.8, edgecolor="white")
axes[0].set_title(f"Volatilidad realizada media {VENTANA_LARGA}h por empresa\n(2024–2026, anualizada)", fontsize=10)
axes[0].set_ylabel(f"Vol. realizada {VENTANA_LARGA}h media")

ret_abs = df.groupby("ticker")["abs_return"].mean().sort_values(ascending=False)
axes[1].bar(ret_abs.index, ret_abs.values,
            color=[COLORES[t] for t in ret_abs.index], alpha=0.8, edgecolor="white")
axes[1].set_title("Retorno absoluto medio horario por empresa\n(2024–2026)", fontsize=10)
axes[1].set_ylabel("Retorno absoluto medio (1h)")

plt.suptitle("Resumen de riesgo intradía — Magnificent 7", fontsize=12, y=1.01)
plt.tight_layout()
guardar_figura("intraday_estadisticas.png")


# ── 6. Guardar el dataset limpio ──────────────────────────────────────────────
# Exportamos con el índice (datetime_et) incluido y formato ISO con zona horaria.

separador("6. GUARDAR DATASET LIMPIO")

os.makedirs(os.path.dirname(RUTA_CLEAN), exist_ok=True)
df.to_csv(RUTA_CLEAN, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
print(f"  Guardado: {RUTA_CLEAN}")
print(f"  Shape   : {df.shape}")
print(f"  Columnas: {df.columns.tolist()}")


# ── 7. Resumen final ──────────────────────────────────────────────────────────
# Imprimimos todo lo que hemos hecho: datos de entrada, transformaciones,
# datos de salida y la lista de figuras generadas.

separador("7. RESUMEN DE LA LIMPIEZA")

print(f"""
  DATOS DE ENTRADA
  ─────────────────────────────────────────────
  Archivo          : {RUTA_RAW}
  Barras           : {len(df_raw):,}
  Columnas         : {len(df_raw.columns)}
  Periodo          : {df_raw.index.min()} → {df_raw.index.max()}

  TRANSFORMACIONES APLICADAS
  ─────────────────────────────────────────────
  1. Duplicados eliminados         : 0 (verificado)
  2. Incoherencias OHLCV           : 0 (verificado)
  3. Primeras barras por ticker    : {len(TICKERS)} eliminadas (1/ticker, NaN log_return_1h inevitable)
  4. Zona horaria                  : normalizada a America/New_York (ET)
  5. NaN vol_realized_{VENTANA_CORTA}h           : {df[f'vol_realized_{VENTANA_CORTA}h'].isna().sum()} → CONSERVADOS (ventana rolling {VENTANA_CORTA}h)
  6. NaN vol_realized_{VENTANA_LARGA}h          : {df[f'vol_realized_{VENTANA_LARGA}h'].isna().sum()} → CONSERVADOS (ventana rolling {VENTANA_LARGA}h)

  DATOS DE SALIDA
  ─────────────────────────────────────────────
  Archivo          : {RUTA_CLEAN}
  Barras           : {len(df):,}
  Columnas         : {len(df.columns)}
  Periodo          : {df.index.min()} → {df.index.max()}

  FIGURAS GENERADAS
  ─────────────────────────────────────────────
  docs/figuras/intraday_nulos_antes.png
  docs/figuras/intraday_nulos_despues.png
  docs/figuras/intraday_retornos_boxplot.png
  docs/figuras/intraday_volatilidad_serie.png
  docs/figuras/intraday_patron_horario.png
  docs/figuras/intraday_correlacion_volatilidad.png
  docs/figuras/intraday_estadisticas.png
""")
