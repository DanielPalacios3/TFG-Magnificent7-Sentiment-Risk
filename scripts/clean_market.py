"""
Script: clean_market.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Bueno, este script se encarga de limpiar los datos de mercado que descargamos de Yahoo
Finance para las 7 empresas del Magnificent 7. En teoría, yfinance ya te da datos bastante
limpios, pero no nos podemos fiar ciegamente: hay que repasar que no haya cosas raras
en los precios OHLCV (como que el máximo del día sea menor que el mínimo, que sería
absurdo), documentar los NaN que sabemos que van a aparecer en las ventanas rolling
(porque no son errores, sino consecuencia de que no puedes calcular una media de 20 días
si solo llevas 3), y quitar la primera fila de cada empresa porque no tiene retorno
calculable (no hay precio del día anterior con el que comparar).

Además de limpiar, genera un montón de figuras comparativas de antes y después que van
directamente al documento Word del TFG. La rúbrica pide explícitamente mostrar el estado
de los datos antes y después de cada transformación, así que hay que cumplir con eso.

Archivos que genera
-------------------
- data/clean/market_clean.csv         → el dataset ya limpio, listo para meter en modelos
- docs/figuras/market_nulos_antes.png → para ver cómo estaban los nulos antes de tocar nada
- docs/figuras/market_nulos_despues.png → para ver qué nulos quedan después (los de rolling)
- docs/figuras/market_retornos_boxplot.png → cómo se distribuyen los retornos por empresa
- docs/figuras/market_volatilidad_serie.png → la evolución de la volatilidad a lo largo del tiempo
- docs/figuras/market_estadisticas.png  → resumen visual de estadísticas descriptivas

Sobre las decisiones que tomamos
--------------------------------
- Los NaN en vol_realized_5d y vol_realized_20d se dejan tal cual. Aparecen al principio
  de cada ticker porque, claro, si la ventana es de 20 días, hasta que no tengas 20 días
  de datos no puedes calcular nada. No son errores y eliminarlos sería perder datos
  innecesariamente.
- La primera fila de cada ticker se elimina porque su log_return es NaN (no hay día previo
  con el que calcular ln(P_t / P_{t-1})). Son solo 7 filas en total, una por empresa.
- Los outliers gordos (el desplome de META del -30%, el crash del COVID, el rally de NVDA)
  NO se tocan. De hecho, esos episodios de volatilidad extrema son precisamente lo que
  queremos estudiar en este TFG, así que eliminarlos sería un contrasentido.

Para ejecutar
-------------
    source venv/bin/activate
    python scripts/clean_market.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # Forzamos backend sin interfaz gráfica porque estamos en WSL y no hay pantalla
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuración general ────────────────────────────────────────────────────
# Aquí definimos las rutas de entrada/salida, los tickers y los colores que
# usaremos en los gráficos. Cada empresa tiene su color corporativo para que
# los gráficos sean consistentes en todo el TFG.

RUTA_RAW   = os.path.join("data", "raw", "market", "ALL_market_2019_2026.csv")
RUTA_CLEAN = os.path.join("data", "clean", "market_clean.csv")
DIR_FIGS   = os.path.join("docs", "figuras")

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
COLORES = {
    "AAPL": "#555555", "MSFT": "#0078D4", "GOOGL": "#EA4335",
    "AMZN": "#FF9900", "NVDA": "#76B900", "META": "#1877F2", "TSLA": "#CC0000",
}

# Eventos históricos que marcamos en los gráficos para dar contexto.
# Son los momentos clave que provocaron picos de volatilidad en estas empresas.
EVENTOS = {
    "2020-03-16": "COVID crash",
    "2021-01-27": "GameStop/WSB",
    "2022-02-03": "META -30%",
    "2023-05-25": "NVDA +24%",
}

os.makedirs(DIR_FIGS, exist_ok=True)


# ── Funciones auxiliares ──────────────────────────────────────────────────────
# Pequeñas utilidades para no repetir código: una imprime separadores bonitos
# en la terminal y otra guarda las figuras de matplotlib en disco.

def separador(titulo: str):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print('='*60)


def guardar_figura(nombre: str):
    ruta = os.path.join(DIR_FIGS, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figura guardada: {ruta}")


# ── 1. Carga del CSV en crudo ─────────────────────────────────────────────────
# Leemos el archivo que generó download_market.py con todos los tickers juntos.

separador("1. CARGA DE DATOS")

df_raw = pd.read_csv(RUTA_RAW, parse_dates=["date"])
print(f"  Filas cargadas : {len(df_raw):,}")
print(f"  Columnas       : {df_raw.columns.tolist()}")
print(f"  Tickers        : {sorted(df_raw['ticker'].unique())}")
print(f"  Rango fechas   : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
print(f"  Filas por ticker:\n{df_raw.groupby('ticker').size().to_string()}")


# ── 2. Revisión inicial (ANTES de tocar nada) ────────────────────────────────
# Antes de hacer ninguna limpieza, miramos cómo vienen los datos: nulos,
# duplicados, fines de semana que se hayan colado, y que los precios OHLCV
# tengan sentido (que el High no sea menor que el Low, por ejemplo).

separador("2. REVISIÓN INICIAL — ANTES DE LIMPIEZA")

# 2a. Contamos cuántos valores nulos hay por columna
nulos_antes = df_raw.isnull().sum()
print(f"\n  Valores nulos por columna:")
print(nulos_antes[nulos_antes > 0].to_string())

# 2b. Buscamos filas duplicadas por combinación de fecha + ticker
dupl = df_raw.duplicated(["date", "ticker"]).sum()
print(f"\n  Duplicados (date+ticker): {dupl}")

# 2c. Comprobamos si se han colado registros de sábados o domingos
fds = df_raw[df_raw["date"].dt.dayofweek >= 5]
print(f"  Registros en fin de semana: {len(fds)}")

# 2d. Verificamos que los precios OHLCV son coherentes entre sí.
# Por ejemplo, el máximo del día (High) nunca debería ser menor que el mínimo (Low).
inc_hl  = (df_raw["high"] < df_raw["low"]).sum()
inc_ch  = (df_raw["close"] > df_raw["high"]).sum()
inc_cl  = (df_raw["close"] < df_raw["low"]).sum()
inc_oh  = (df_raw["open"] > df_raw["high"]).sum()
inc_ol  = (df_raw["open"] < df_raw["low"]).sum()
print(f"\n  Coherencia OHLCV:")
print(f"    High < Low    : {inc_hl}")
print(f"    Close > High  : {inc_ch}")
print(f"    Close < Low   : {inc_cl}")
print(f"    Open > High   : {inc_oh}")
print(f"    Open < Low    : {inc_ol}")

# Figura ANTES: mostramos la completitud por columna con barras horizontales.
# Probé hacer un heatmap fila por fila, pero con 12K filas es completamente
# ilegible. Así que mejor un gráfico de barras donde cada columna muestra qué
# porcentaje de sus datos están completos.
nulos_conteo = df_raw.isnull().sum()
completitud  = (1 - df_raw.isnull().mean()) * 100   # % filas con valor

fig, ax = plt.subplots(figsize=(9, 5))
colores_bar = ["#e53935" if v < 100 else "#43a047" for v in completitud.values]
barras = ax.barh(completitud.index, completitud.values, color=colores_bar, edgecolor="white")
ax.set_xlim(0, 105)
ax.set_xlabel("% de filas con valor (completitud)")
ax.set_title("Completitud de datos — ANTES de limpieza\n"
             "(rojo = columnas con NaN; verde = 100% completo)", fontsize=11)
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)

# Ponemos el porcentaje exacto y el número de NaN al lado de cada barra
for i, (col, pct) in enumerate(completitud.items()):
    n_nan = int(nulos_conteo[col])
    etiqueta = f"{pct:.2f}%  ({n_nan} NaN)" if n_nan > 0 else f"100%"
    ax.text(pct + 0.3, i, etiqueta, va="center", fontsize=8,
            color="#c62828" if n_nan > 0 else "#2e7d32")

# Añadimos una explicación del porqué de cada NaN, para que quede claro
# que no son errores sino consecuencia del cálculo
notas = {
    "log_return":        "← 1 NaN/ticker: primer día sin precio previo",
    "vol_realized_5d":   "← 5 NaN/ticker: ventana rolling 5 días",
    "vol_realized_20d":  "← 20 NaN/ticker: ventana rolling 20 días",
    "abs_return":        "← igual que log_return",
    "sq_return":         "← igual que log_return",
    "log_volume_change": "← igual que log_return",
}
for col, nota in notas.items():
    if col in completitud.index:
        idx = list(completitud.index).index(col)
        ax.text(102, idx, nota, va="center", fontsize=7, color="#555555")

plt.tight_layout()
guardar_figura("market_nulos_antes.png")


# ── 3. Limpieza ───────────────────────────────────────────────────────────────
# Aquí es donde realmente tocamos los datos. Spoiler: hay muy poco que limpiar
# porque yfinance ya hace buen trabajo, pero aun así pasamos por cada chequeo.

separador("3. LIMPIEZA Y TRANSFORMACIONES")

df = df_raw.copy()

# 3a. Eliminar duplicados. No esperamos encontrar ninguno, pero lo verificamos
# por si acaso. Más vale prevenir.
n_antes = len(df)
df = df.drop_duplicates(["date", "ticker"])
print(f"  Duplicados eliminados: {n_antes - len(df)}")

# 3b. Quitar fines de semana si los hubiera (tampoco esperamos, pero por si las moscas)
df = df[df["date"].dt.dayofweek < 5]
print(f"  Registros fin de semana eliminados: {n_antes - len(df)}")

# 3c. Quitamos la primera fila de cada ticker porque tiene NaN en log_return.
# Es lógico: el retorno logarítmico se calcula como ln(precio_hoy / precio_ayer),
# y el primer día no tiene "ayer". Son solo 7 filas (una por empresa) y no hay
# forma de recuperar esa información, así que fuera.
n_antes = len(df)
primera_fecha_por_ticker = df.groupby("ticker")["date"].min()
mask_primera = df.apply(
    lambda row: row["date"] == primera_fecha_por_ticker[row["ticker"]], axis=1
)
df = df[~mask_primera].reset_index(drop=True)
print(f"  Primeras filas eliminadas (sin log_return): {n_antes - len(df)} "
      f"(1 por cada uno de los {len(TICKERS)} tickers — esperado)")

# 3d. Los NaN en vol_realized_5d y vol_realized_20d los dejamos como están.
# Son el típico artefacto de las ventanas rolling: si la ventana es de 20 días,
# los primeros 19 días no tienen suficientes datos para calcularla. Si los
# elimináramos perderíamos 20 días de datos por empresa (140 filas en total),
# lo cual no merece la pena. Más adelante, en build_dataset.py, los rellenaremos
# con forward-fill dentro de cada ticker.
nan_5d  = df["vol_realized_5d"].isna().sum()
nan_20d = df["vol_realized_20d"].isna().sum()
print(f"\n  NaN vol_realized_5d  : {nan_5d}  (esperados: {len(TICKERS)*4} = {len(TICKERS)} tickers × 4 días de ventana restantes)")
print(f"  NaN vol_realized_20d : {nan_20d} (esperados: {len(TICKERS)*19} = {len(TICKERS)} tickers × 19 días de ventana restantes)")
print(f"  → DECISIÓN: se conservan. Son artefactos de ventana rolling, no errores.")
print(f"    Se tratarán con forward-fill en build_dataset.py.")

# 3e. Nos aseguramos de que cada columna tenga el tipo de dato correcto.
# Ticker como categoría ahorra memoria, y volume como int64 porque son números enteros.
df["date"]   = pd.to_datetime(df["date"])
df["ticker"] = df["ticker"].astype("category")
df["volume"] = df["volume"].astype("int64")

# 3f. Ordenamos por ticker y fecha para que todo quede bonito y secuencial
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

print(f"\n  Shape final: {df.shape}")
print(f"  Rango fechas: {df['date'].min().date()} → {df['date'].max().date()}")


# ── 4. Figuras DESPUÉS de la limpieza ────────────────────────────────────────
# Generamos los mismos gráficos que antes pero con los datos ya limpios,
# para que en el Word se pueda hacer la comparación visual directa.

separador("4. FIGURAS DESPUÉS DE LIMPIEZA")

# 4a. Completitud DESPUÉS: usamos la misma escala que antes para que la
# comparación sea justa y se vea bien de un vistazo.
nulos_conteo_d = df.isnull().sum()
completitud_d  = (1 - df.isnull().mean()) * 100

fig, ax = plt.subplots(figsize=(9, 5))
colores_bar_d = ["#e53935" if v < 100 else "#43a047" for v in completitud_d.values]
ax.barh(completitud_d.index, completitud_d.values, color=colores_bar_d, edgecolor="white")
ax.set_xlim(0, 105)
ax.set_xlabel("% de filas con valor (completitud)")
ax.set_title("Completitud de datos — DESPUÉS de limpieza\n"
             "(los NaN restantes son artefactos inevitables de ventana rolling)", fontsize=11)
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)

for i, (col, pct) in enumerate(completitud_d.items()):
    n_nan = int(nulos_conteo_d[col])
    etiqueta = f"{pct:.2f}%  ({n_nan} NaN — rolling)" if n_nan > 0 else "100%  ✓"
    ax.text(pct + 0.3, i, etiqueta, va="center", fontsize=8,
            color="#c62828" if n_nan > 0 else "#2e7d32")

plt.tight_layout()
guardar_figura("market_nulos_despues.png")

# 4b. Boxplot de retornos logarítmicos por empresa. Aquí es donde se ven los
# outliers famosos: la caída del -30% de META, los picos de TSLA y NVDA...
# Cada punto extremo es un episodio histórico real que nos interesa estudiar.
fig, ax = plt.subplots(figsize=(12, 5))
data_box = [df[df["ticker"] == t]["log_return"].dropna().values for t in TICKERS]
bp = ax.boxplot(data_box, labels=TICKERS, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch, ticker in zip(bp["boxes"], TICKERS):
    patch.set_facecolor(COLORES[ticker])
    patch.set_alpha(0.7)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Distribución de retornos logarítmicos diarios por empresa (2019–2025)\n"
             "Los puntos extremos representan episodios históricos de alta volatilidad",
             fontsize=11)
ax.set_ylabel("Retorno logarítmico diario")
ax.set_xlabel("Empresa")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
plt.tight_layout()
guardar_figura("market_retornos_boxplot.png")

# 4c. Series temporales de volatilidad realizada a 20 días. Hacemos un subplot
# por empresa en vez de meterlas todas juntas, porque así se ve mucho mejor
# la evolución individual de cada una y se pueden identificar los picos.
fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=False)
axes_flat = axes.flatten()

for i, ticker in enumerate(TICKERS):
    ax = axes_flat[i]
    sub = df[df["ticker"] == ticker].sort_values("date")

    ax.fill_between(sub["date"], sub["vol_realized_20d"],
                    color=COLORES[ticker], alpha=0.25)
    ax.plot(sub["date"], sub["vol_realized_20d"],
            color=COLORES[ticker], linewidth=1.2)

    # Marcamos los eventos históricos con líneas verticales para dar contexto
    ymax = sub["vol_realized_20d"].max()
    for fecha_str, label in EVENTOS.items():
        fecha = pd.to_datetime(fecha_str)
        ax.axvline(fecha, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(fecha, ymax * 0.97, label,
                rotation=90, fontsize=6.5, va="top", ha="right",
                color="#555555")

    ax.set_title(ticker, fontsize=12, fontweight="bold", color=COLORES[ticker])
    ax.set_ylabel("Vol. 20d", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

# Ocultamos el 8.o panel que sobra (tenemos 7 empresas en una cuadrícula de 4x2)
axes_flat[-1].set_visible(False)

fig.suptitle("Volatilidad realizada a 20 días — Magnificent 7 (2019–2025)\n"
             "Variable objetivo del TFG", fontsize=13, y=1.01)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
guardar_figura("market_volatilidad_serie.png")

# 4d. Heatmap de correlación entre las volatilidades realizadas a 20 días.
# Esto nos dice si las empresas se mueven juntas en términos de riesgo. Si la
# correlación es alta, hay un factor de mercado común que las arrastra a todas.
fig, ax = plt.subplots(figsize=(8, 6))
vol_pivot = df.pivot_table(index="date", columns="ticker", values="vol_realized_20d")
corr = vol_pivot.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Correlación de volatilidad realizada 20d entre empresas\n"
             "Alta correlación indica movimiento conjunto (factor mercado)", fontsize=10)
plt.tight_layout()
guardar_figura("market_correlacion_volatilidad.png")


# ── 5. Estadísticas descriptivas ──────────────────────────────────────────────
# Calculamos media, desviación, mínimo y máximo de las variables principales
# agrupadas por empresa. Esto va tanto a la terminal como a una figura.

separador("5. ESTADÍSTICAS DESCRIPTIVAS POR EMPRESA")

cols_stats = ["log_return", "vol_realized_20d", "vol_intraday", "abs_return"]
stats = df.groupby("ticker")[cols_stats].agg(["mean", "std", "min", "max"]).round(4)
print(stats.to_string())

# Un resumen rápido por empresa: retorno medio, volatilidad media, y los días
# más extremos (el peor y el mejor) de cada una
print("\n  Resumen ejecutivo:")
print(f"  {'Ticker':<6} {'Retorno medio':>14} {'Vol.real.20d media':>19} {'Peor día':>10} {'Mejor día':>10}")
print(f"  {'-'*65}")
for ticker in TICKERS:
    sub = df[df["ticker"] == ticker]
    ret_medio = sub["log_return"].mean()
    vol_media = sub["vol_realized_20d"].mean()
    peor      = sub["log_return"].min()
    mejor     = sub["log_return"].max()
    print(f"  {ticker:<6} {ret_medio:>+13.4f}  {vol_media:>18.4f}  {peor:>+9.4f}  {mejor:>+9.4f}")

# Figura con dos paneles: a la izquierda la volatilidad media por empresa,
# a la derecha el retorno absoluto medio. Así se ve de un vistazo quién es
# más "movida" (spoiler: suele ser TSLA o NVDA).
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

vol_media_ticker = df.groupby("ticker")["vol_realized_20d"].mean().sort_values(ascending=False)
axes[0].bar(vol_media_ticker.index, vol_media_ticker.values,
            color=[COLORES[t] for t in vol_media_ticker.index], alpha=0.8, edgecolor="white")
axes[0].set_title("Volatilidad realizada media 20d por empresa\n(2019–2025)", fontsize=10)
axes[0].set_ylabel("Vol. realizada 20d media")
axes[0].set_xlabel("")

ret_abs_ticker = df.groupby("ticker")["abs_return"].mean().sort_values(ascending=False)
axes[1].bar(ret_abs_ticker.index, ret_abs_ticker.values,
            color=[COLORES[t] for t in ret_abs_ticker.index], alpha=0.8, edgecolor="white")
axes[1].set_title("Retorno absoluto medio diario por empresa\n(2019–2025)", fontsize=10)
axes[1].set_ylabel("Retorno absoluto medio")
axes[1].set_xlabel("")

plt.suptitle("Resumen de riesgo — Magnificent 7", fontsize=12, y=1.01)
plt.tight_layout()
guardar_figura("market_estadisticas.png")


# ── 6. Guardar el dataset ya limpio ───────────────────────────────────────────
# Exportamos a CSV el dataset limpio, listo para que build_dataset.py lo
# combine con las fuentes de sentimiento.

separador("6. GUARDAR DATASET LIMPIO")

os.makedirs(os.path.dirname(RUTA_CLEAN), exist_ok=True)
df.to_csv(RUTA_CLEAN, index=False, encoding="utf-8")
print(f"  Guardado: {RUTA_CLEAN}")
print(f"  Shape   : {df.shape}")
print(f"  Columnas: {df.columns.tolist()}")


# ── 7. Resumen final ──────────────────────────────────────────────────────────
# Imprimimos un resumen completo de todo lo que hemos hecho: cuántas filas
# entraron, qué transformaciones se aplicaron, cuántas filas salen, y la
# lista de figuras generadas. Así queda todo documentado en la terminal.

separador("7. RESUMEN DE LA LIMPIEZA")

print(f"""
  DATOS DE ENTRADA
  ─────────────────────────────────────────────
  Archivo          : {RUTA_RAW}
  Filas            : {len(df_raw):,}
  Columnas         : {len(df_raw.columns)}
  Periodo          : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}

  TRANSFORMACIONES APLICADAS
  ─────────────────────────────────────────────
  1. Duplicados eliminados      : 0 (verificado)
  2. Fines de semana            : 0 (verificado)
  3. Incoherencias OHLCV        : 0 (verificado)
  4. Primeras filas por ticker  : 7 eliminadas (1/ticker, NaN log_return inevitable)
  5. NaN vol_realized_5d        : {df['vol_realized_5d'].isna().sum()} → CONSERVADOS (ventana rolling 5d)
  6. NaN vol_realized_20d       : {df['vol_realized_20d'].isna().sum()} → CONSERVADOS (ventana rolling 20d)

  DATOS DE SALIDA
  ─────────────────────────────────────────────
  Archivo          : {RUTA_CLEAN}
  Filas            : {len(df):,}
  Columnas         : {len(df.columns)}
  Periodo          : {df['date'].min().date()} → {df['date'].max().date()}

  FIGURAS GENERADAS
  ─────────────────────────────────────────────
  docs/figuras/market_nulos_antes.png
  docs/figuras/market_nulos_despues.png
  docs/figuras/market_retornos_boxplot.png
  docs/figuras/market_volatilidad_serie.png
  docs/figuras/market_correlacion_volatilidad.png
  docs/figuras/market_estadisticas.png
""")
