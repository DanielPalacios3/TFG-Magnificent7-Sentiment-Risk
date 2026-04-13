"""
Script: clean_news.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Este script se encarga de limpiar y agregar los datos de noticias GDELT que descargamos
de BigQuery. GDELT es una base de datos gigante de noticias globales que, entre otras
cosas, calcula automáticamente el "tono emocional" de cada artículo (V2Tone). Eso nos
ahorra tener que pasar todos los artículos por un modelo NLP en esta fase.

El script produce dos cosas:
  1. Una agregación diaria por ticker (media del tono, dispersión, volumen de noticias...),
     que es lo que alimenta directamente a los modelos GARCH-X y XGBoost.
  2. Los artículos individuales ya limpios, por si luego queremos pasarlos por FinBERT
     en la Fase 3.

Sobre por qué usamos GDELT y no otra fuente: básicamente es la única opción gratuita
que cubre todo el periodo desde 2019. Probamos con Finnhub (solo un año), Stock News API
(sin histórico real) y la API REST de GDELT (solo 3 meses). BigQuery fue la solución.

Limitaciones que hay que tener en cuenta
-----------------------------------------
- Solo tenemos un componente de V2Tone: el tono general. El formato completo tiene 7
  componentes (densidad positiva, negativa, polaridad, etc.), pero en la consulta BigQuery
  solo extrajimos el primero con SPLIT(V2Tone, ',')[OFFSET(0)]. Para lo que necesitamos
  (que es lo mismo que usó Tetlock en 2007) el tono general es suficiente.
- No filtramos por idioma en la descarga, así que hay artículos en chino, árabe, etc.
  GDELT aplica el mismo algoritmo léxico a todos los idiomas, así que asumimos que el
  tono es razonablemente comparable. No es perfecto, pero es lo que hay con los datos
  que tenemos.

Archivos que genera
-------------------
- data/clean/news_clean_daily.csv    → una fila por (fecha, ticker) con tono agregado
- data/clean/news_clean_articles.csv → artículos individuales para FinBERT
- docs/figuras/news_nulos_antes.png
- docs/figuras/news_nulos_despues.png
- docs/figuras/news_tono_boxplot.png
- docs/figuras/news_tono_temporal.png
- docs/figuras/news_volumen_noticias.png
- docs/figuras/news_correlacion_tono.png

Para ejecutar
-------------
    source venv/bin/activate
    python scripts/clean_news.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuración general ────────────────────────────────────────────────────
# Rutas, tickers, colores y los límites del tono GDELT. V2Tone va de -100 a
# +100; cualquier valor fuera de ese rango es un error de parseo.

DIR_NEWS       = os.path.join("data", "raw", "news")
RUTA_DAILY     = os.path.join("data", "clean", "news_clean_daily.csv")
RUTA_ARTICLES  = os.path.join("data", "clean", "news_clean_articles.csv")
DIR_FIGS       = os.path.join("docs", "figuras")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
COLORES = {
    "AAPL": "#555555", "MSFT": "#0078D4", "GOOGL": "#EA4335",
    "AMZN": "#FF9900", "NVDA": "#76B900", "META": "#1877F2", "TSLA": "#CC0000",
}

# El tono de GDELT V2Tone va de -100 (muy negativo) a +100 (muy positivo).
# Si encontramos algo fuera de este rango, seguro que es un error de parseo.
TONE_MIN = -100.0
TONE_MAX = 100.0

# Periodo temporal del estudio
FECHA_INICIO = "2019-01-01"
FECHA_FIN    = "2026-04-02"

os.makedirs(DIR_FIGS, exist_ok=True)
os.makedirs("data/clean", exist_ok=True)


# ── Funciones auxiliares ──────────────────────────────────────────────────────
# Utilidades para separadores en terminal, guardar figuras y cargar CSVs.

def separador(titulo: str):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print('='*60)


def guardar_figura(nombre: str):
    ruta = os.path.join(DIR_FIGS, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figura guardada: {ruta}")


def cargar_ticker(ticker: str) -> pd.DataFrame:
    """Carga el CSV de noticias de un ticker. Intenta primero el archivo con
    sufijo _2026 y si no existe cae al _2025. Solo lee las columnas que
    necesitamos para no cargar datos innecesarios en memoria."""
    ruta_26 = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2026.csv")
    ruta = ruta_26 if os.path.exists(ruta_26) else os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2025.csv")
    df = pd.read_csv(
        ruta,
        usecols=["fecha", "url", "domain", "tone", "query_ticker"],
        dtype={"url": str, "domain": str, "query_ticker": str},
        low_memory=False,
    )
    return df


# ── 1. Revisión inicial (ANTES de tocar nada) ────────────────────────────────
# Cargamos cada ticker y miramos cuántos artículos hay, cuántos nulos en tono
# y cuántas URLs duplicadas. Esto nos da una idea del volumen de limpieza
# que nos espera.

separador("1. REVISIÓN INICIAL — TODOS LOS TICKERS")

stats_antes = {}
for ticker in TICKERS:
    df = cargar_ticker(ticker)
    stats_antes[ticker] = {
        "filas_raw":    len(df),
        "nulos_tone":   df["tone"].isna().sum(),
        "nulos_fecha":  df["fecha"].isna().sum(),
        "nulos_url":    df["url"].isna().sum(),
        "duplicados_url": df["url"].duplicated().sum(),
    }
    print(f"  {ticker}: {len(df):>10,} filas | "
          f"nulos tone: {stats_antes[ticker]['nulos_tone']:>5} | "
          f"dup URL: {stats_antes[ticker]['duplicados_url']:>6,}")

total_raw = sum(v["filas_raw"] for v in stats_antes.values())
print(f"\n  TOTAL artículos en crudo: {total_raw:,}")


# Figura ANTES: mostramos cuántos artículos tiene cada empresa antes de limpiar
fig, ax = plt.subplots(figsize=(10, 4))
tickers_labels = list(stats_antes.keys())
filas_raw = [stats_antes[t]["filas_raw"] for t in tickers_labels]
bars = ax.bar(tickers_labels, filas_raw,
              color=[COLORES[t] for t in tickers_labels], alpha=0.85, edgecolor="white")
for bar, n in zip(bars, filas_raw):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000,
            f"{n/1e6:.2f}M", ha="center", va="bottom", fontsize=9)
ax.set_title("Volumen de artículos GDELT por empresa — ANTES de limpieza\n"
             "(todos los idiomas, sin deduplicar)", fontsize=11)
ax.set_ylabel("Número de artículos")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
plt.tight_layout()
guardar_figura("news_nulos_antes.png")


# ── 2. Limpieza ticker por ticker ─────────────────────────────────────────────
# Procesamos cada empresa por separado: parseamos fechas, quitamos nulos de
# tono, eliminamos duplicados por URL y al final agregamos por día.

separador("2. LIMPIEZA Y TRANSFORMACIONES")

articulos_limpios = []   # para news_clean_articles.csv
diarios_por_ticker = []  # para news_clean_daily.csv
stats_despues = {}

for ticker in TICKERS:
    print(f"\n  [{ticker}] Procesando...")
    df = cargar_ticker(ticker)
    n0 = len(df)

    # 2a. Convertimos la fecha a datetime y nos quedamos solo con el periodo
    # del estudio. Los artículos fuera de rango no nos sirven.
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    n_fecha_invalida = df["fecha"].isna().sum()
    df = df.dropna(subset=["fecha"])
    df = df[(df["fecha"] >= FECHA_INICIO) & (df["fecha"] <= FECHA_FIN)]
    n1 = len(df)

    # 2b. Quitamos artículos sin tono o con tono fuera de rango [-100, 100].
    # Si un artículo no tiene tono, no nos sirve para nada (el tono es
    # literalmente la variable de sentimiento). Y si está fuera de rango,
    # algo fue mal en el parseo de BigQuery.
    df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
    n_tone_nulo = df["tone"].isna().sum()
    n_tone_fuera_rango = ((df["tone"] < TONE_MIN) | (df["tone"] > TONE_MAX)).sum()
    df = df.dropna(subset=["tone"])
    df = df[(df["tone"] >= TONE_MIN) & (df["tone"] <= TONE_MAX)]
    n2 = len(df)

    # 2c. Eliminamos duplicados por URL dentro del mismo ticker. GDELT a veces
    # indexa el mismo artículo varias veces porque lo captura desde distintas
    # fuentes de noticias. La URL identifica de forma única cada artículo.
    n_dup = df["url"].duplicated().sum()
    df = df.drop_duplicates(subset=["url"])
    n3 = len(df)

    # 2d. Si la URL es nula, no podemos identificar el artículo ni deduplicar,
    # así que fuera
    df = df.dropna(subset=["url"])
    n4 = len(df)

    stats_despues[ticker] = {
        "filas_raw":          n0,
        "eliminadas_fecha":   n0 - n1,
        "eliminadas_tone":    n1 - n2,
        "eliminadas_dup":     n_dup,
        "filas_limpias":      n4,
        "pct_conservado":     n4 / n0 * 100,
    }

    print(f"    Raw: {n0:>10,} → "
          f"fecha inválida: -{n0-n1:>5,} | "
          f"tone nulo/rango: -{n1-n2:>5,} | "
          f"duplicados: -{n_dup:>6,} → "
          f"Limpio: {n4:>9,} ({n4/n0*100:.1f}%)")

    # Guardamos los artículos limpios por si luego queremos pasarlos por FinBERT
    df_art = df[["fecha", "url", "domain", "tone", "query_ticker"]].copy()
    df_art["fecha"] = df_art["fecha"].dt.date
    articulos_limpios.append(df_art)

    # 2e. Ahora viene la agregación diaria, que es lo importante para los modelos.
    # Para cada día y ticker calculamos varias métricas:
    #   - tone_mean: el tono promedio del día, que es la señal principal para GARCH-X
    #   - tone_std: la dispersión del tono (si es alta, las fuentes no se ponen de acuerdo)
    #   - tone_min/max: los extremos del día
    #   - n_noticias: cuántos artículos hubo (la cobertura mediática en sí es información)
    #   - pct_negativo/positivo: qué porcentaje de los artículos son negativos o positivos
    df["fecha_solo"] = df["fecha"].dt.date

    diario = df.groupby("fecha_solo").agg(
        tone_mean    = ("tone", "mean"),
        tone_std     = ("tone", "std"),
        tone_min     = ("tone", "min"),
        tone_max     = ("tone", "max"),
        tone_median  = ("tone", "median"),
        n_noticias   = ("tone", "count"),
        pct_negativo = ("tone", lambda x: (x < 0).mean() * 100),
        pct_positivo = ("tone", lambda x: (x > 0).mean() * 100),
    ).reset_index()

    diario.rename(columns={"fecha_solo": "fecha"}, inplace=True)
    diario["fecha"]  = pd.to_datetime(diario["fecha"])
    diario["ticker"] = ticker
    diarios_por_ticker.append(diario)

    print(f"    Días únicos con noticias: {len(diario):,} | "
          f"tone_mean: {diario['tone_mean'].mean():.3f} ± {diario['tone_mean'].std():.3f}")


# ── 3. Juntamos todo en DataFrames consolidados ──────────────────────────────
# Concatenamos los artículos limpios y las agregaciones diarias de los 7 tickers.

separador("3. CONSOLIDACIÓN")

df_articles = pd.concat(articulos_limpios, ignore_index=True)
df_daily    = pd.concat(diarios_por_ticker, ignore_index=True)
df_daily    = df_daily.sort_values(["ticker", "fecha"]).reset_index(drop=True)

total_limpio = sum(v["filas_limpias"] for v in stats_despues.values())
print(f"\n  Artículos totales limpios : {total_limpio:,}")
print(f"  Reducción total           : {total_raw - total_limpio:,} "
      f"({(total_raw - total_limpio)/total_raw*100:.1f}%)")
print(f"\n  Agregación diaria:")
print(f"  Filas en news_clean_daily : {len(df_daily):,}")
print(f"  Rango fechas              : {df_daily['fecha'].min().date()} → "
      f"{df_daily['fecha'].max().date()}")
print(f"\n  Por ticker:")
for ticker in TICKERS:
    sub = df_daily[df_daily["ticker"] == ticker]
    print(f"    {ticker}: {len(sub):>4,} días | "
          f"tone_mean={sub['tone_mean'].mean():+.3f} | "
          f"n_noticias_media={sub['n_noticias'].mean():.0f}/día")


# ── 4. Figuras ────────────────────────────────────────────────────────────────
# Generamos varias visualizaciones: comparación antes/después, distribución
# del tono, evolución temporal, volumen de noticias y correlaciones.

separador("4. FIGURAS")

# 4a. Comparamos cuántos artículos teníamos antes y después de limpiar.
# Las barras grises son el antes y las de color el después, con el porcentaje
# de conservación encima de cada una.
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(TICKERS))
w = 0.38
bars_antes  = ax.bar(x - w/2, [stats_antes[t]["filas_raw"]      for t in TICKERS],
                     w, label="Antes", color="#90A4AE", edgecolor="white")
bars_despues = ax.bar(x + w/2, [stats_despues[t]["filas_limpias"] for t in TICKERS],
                      w, label="Después", color=[COLORES[t] for t in TICKERS],
                      edgecolor="white", alpha=0.9)
for bar, t in zip(bars_despues, TICKERS):
    pct = stats_despues[t]["pct_conservado"]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(TICKERS)
ax.set_ylabel("Número de artículos")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
ax.set_title("Artículos GDELT antes y después de limpieza por empresa\n"
             "(% = fracción conservada)", fontsize=11)
ax.legend()
plt.tight_layout()
guardar_figura("news_nulos_despues.png")

# 4b. Boxplot del tono diario por empresa. Nos permite ver si alguna empresa
# tiende a tener cobertura mediática más positiva o negativa que las demás.
fig, ax = plt.subplots(figsize=(11, 5))
data_box  = [df_daily[df_daily["ticker"] == t]["tone_mean"].dropna().values for t in TICKERS]
bp = ax.boxplot(data_box, labels=TICKERS, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch, t in zip(bp["boxes"], TICKERS):
    patch.set_facecolor(COLORES[t])
    patch.set_alpha(0.7)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Distribución del tono GDELT diario por empresa (2019–2026)\n"
             "Valores positivos = cobertura favorable, negativos = desfavorable",
             fontsize=11)
ax.set_ylabel("Tono medio diario (V2Tone)")
ax.set_xlabel("Empresa")
plt.tight_layout()
guardar_figura("news_tono_boxplot.png")

# 4c. Evolución del tono mes a mes por empresa. Usamos subplots individuales
# para que se vea bien cada una. Las zonas verdes son meses con tono positivo
# y las rojas con tono negativo, con eventos históricos marcados.
fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True, sharey=False)
axes_flat = axes.flatten()

for i, ticker in enumerate(TICKERS):
    ax = axes_flat[i]
    sub = df_daily[df_daily["ticker"] == ticker].copy()
    sub["mes"] = sub["fecha"].dt.to_period("M").dt.to_timestamp()
    mensual = sub.groupby("mes")["tone_mean"].mean().reset_index()

    ax.fill_between(mensual["mes"], mensual["tone_mean"],
                    where=mensual["tone_mean"] >= 0, alpha=0.4, color="#43a047",
                    interpolate=True, label="Positivo")
    ax.fill_between(mensual["mes"], mensual["tone_mean"],
                    where=mensual["tone_mean"] < 0, alpha=0.4, color="#e53935",
                    interpolate=True, label="Negativo")
    ax.plot(mensual["mes"], mensual["tone_mean"], color=COLORES[ticker], linewidth=1.1)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    # Marcamos eventos históricos con líneas verticales para dar contexto
    for fecha_str, label in [("2020-03-01", "COVID"), ("2022-02-01", "META -30%"),
                               ("2023-05-01", "Boom IA")]:
        ax.axvline(pd.Timestamp(fecha_str), color="#888", linestyle=":", linewidth=0.8)

    ax.set_title(ticker, fontsize=11, fontweight="bold", color=COLORES[ticker])
    ax.set_ylabel("Tono medio", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

axes_flat[-1].set_visible(False)
fig.suptitle("Evolución mensual del tono GDELT — Magnificent 7 (2019–2026)\n"
             "Verde = período de sentimiento positivo | Rojo = negativo",
             fontsize=12, y=1.01)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
guardar_figura("news_tono_temporal.png")

# 4d. Volumen total de noticias por empresa y año. Usamos barras apiladas
# para ver cómo se reparte la cobertura mediática a lo largo de los años.
df_daily["año"] = df_daily["fecha"].dt.year
vol_anual = df_daily.groupby(["año", "ticker"])["n_noticias"].sum().reset_index()
vol_pivot  = vol_anual.pivot(index="año", columns="ticker", values="n_noticias").fillna(0)

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(len(vol_pivot))
for ticker in TICKERS:
    if ticker in vol_pivot.columns:
        ax.bar(vol_pivot.index, vol_pivot[ticker], bottom=bottom,
               label=ticker, color=COLORES[ticker], alpha=0.85, edgecolor="white")
        bottom += vol_pivot[ticker].values
ax.set_title("Volumen total de noticias GDELT por empresa y año (2019–2026)\n"
             "(artículos limpios tras deduplicación)", fontsize=11)
ax.set_ylabel("Número de artículos")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.set_xlabel("Año")
plt.tight_layout()
guardar_figura("news_volumen_noticias.png")

# 4e. Heatmap de correlación del tono diario entre empresas. Si es alta, el
# sentimiento mediático se mueve al unísono (efecto de mercado general).
tone_pivot = df_daily.pivot_table(index="fecha", columns="ticker", values="tone_mean")
corr = tone_pivot.corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Correlación del tono GDELT diario entre empresas\n"
             "Alta correlación indica sentimiento de mercado compartido", fontsize=10)
plt.tight_layout()
guardar_figura("news_correlacion_tono.png")


# ── 5. Guardar los resultados ─────────────────────────────────────────────────
# Exportamos tanto la agregación diaria como los artículos individuales.

separador("5. GUARDAR ARCHIVOS")

# Reordenamos las columnas del daily para que tengan un orden lógico
cols_daily = ["fecha", "ticker", "tone_mean", "tone_std", "tone_median",
              "tone_min", "tone_max", "n_noticias", "pct_negativo", "pct_positivo"]
df_daily[cols_daily].to_csv(RUTA_DAILY, index=False, encoding="utf-8")
print(f"  Guardado: {RUTA_DAILY}  ({len(df_daily):,} filas)")

df_articles.to_csv(RUTA_ARTICLES, index=False, encoding="utf-8")
print(f"  Guardado: {RUTA_ARTICLES}  ({len(df_articles):,} filas)")


# ── 6. Resumen final ──────────────────────────────────────────────────────────
# Imprimimos un resumen completo: cuántos artículos entraron, cuántos se
# eliminaron por cada motivo, las estadísticas del tono y las figuras generadas.

separador("6. RESUMEN FINAL")

print(f"""
  ENTRADA
  ──────────────────────────────────────────────────
  Artículos en crudo (7 CSVs) : {total_raw:>12,}
  Periodo                      : 2019-01-01 → 2025-12-31

  TRANSFORMACIONES
  ──────────────────────────────────────────────────""")
for ticker in TICKERS:
    s = stats_despues[ticker]
    print(f"  {ticker}: -{s['eliminadas_fecha']:>6,} fecha | "
          f"-{s['eliminadas_tone']:>5,} tone | "
          f"-{s['eliminadas_dup']:>7,} dup → {s['filas_limpias']:>9,} ({s['pct_conservado']:.1f}%)")

print(f"""
  SALIDA
  ──────────────────────────────────────────────────
  Artículos limpios (articles) : {len(df_articles):>12,}
  Filas agregación diaria      : {len(df_daily):>12,}

  ESTADÍSTICAS TONO (tone_mean diario)
  ──────────────────────────────────────────────────""")
for ticker in TICKERS:
    sub = df_daily[df_daily["ticker"] == ticker]["tone_mean"]
    print(f"  {ticker}: media={sub.mean():+.3f} | "
          f"std={sub.std():.3f} | "
          f"min={sub.min():.2f} | max={sub.max():.2f}")

print(f"""
  FIGURAS GENERADAS
  ──────────────────────────────────────────────────
  docs/figuras/news_nulos_antes.png
  docs/figuras/news_nulos_despues.png
  docs/figuras/news_tono_boxplot.png
  docs/figuras/news_tono_temporal.png
  docs/figuras/news_volumen_noticias.png
  docs/figuras/news_correlacion_tono.png
""")
