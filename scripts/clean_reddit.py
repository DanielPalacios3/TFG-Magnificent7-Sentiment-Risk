"""
Script: clean_reddit.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Este script limpia los posts de Reddit que descargamos con Arctic Shift para las 7
empresas del Magnificent 7. Al final genera dos cosas:
  1. Una agregación diaria por ticker con métricas como número de posts, score medio,
     porcentaje de posts de WSB, etc. Esto es lo que alimenta a XGBoost.
  2. Los posts individuales ya limpios con un campo texto_completo listo para FinBERT.

Usamos Reddit en vez de Twitter/X porque Elon Musk cerró el acceso gratuito a la API
histórica de Twitter en 2023. Arctic Shift nos da acceso al histórico completo de Reddit
de forma gratuita y sin credenciales, lo cual es ideal. Los subreddits financieros
(r/wallstreetbets, r/stocks, r/investing, r/StockMarket, r/options) son donde se concentra
la mayor parte del sentimiento retail sobre estas empresas.

Decisiones de limpieza (cada una tiene su justificación)
---------------------------------------------------------
1. DUPLICADOS: quitamos duplicados por (id, ticker). Ojo: si un post menciona $AAPL y
   $TSLA, aparece correctamente en ambos tickers y eso NO es un duplicado.
2. BOTS: eliminamos cuentas que hemos identificado manualmente como bots (publican
   resúmenes automáticos diarios, alertas de señales, etc.). Inflarían el volumen sin
   aportar sentimiento real de personas.
3. POSTS [removed]/[deleted]: no eliminamos el post entero. Aunque un moderador haya
   borrado el cuerpo, el título sigue ahí y suele ser informativo. Simplemente marcamos
   el texto como vacío.
4. LONGITUD MINIMA: si texto_completo tiene menos de 10 caracteres, lo descartamos
   porque no hay contenido semántico útil. EXCEPCION: WSB se libra de este filtro
   (ver punto 7).
5. IDIOMA: los 5 subreddits son anglófonos por naturaleza, así que no hace falta filtrar.
6. SCORE <= 0: los conservamos. Un post con 0 upvotes puede ser una opinión minoritaria
   pero sigue siendo una señal válida de sentimiento.
7. TEXTO_COMPLETO: aquí tomamos una decisión importante que depende del subreddit:
   - En r/wallstreetbets el título lo es todo ("TSLA to the moon", "Buy the dip NOW").
     Muchos posts de WSB ni siquiera tienen cuerpo. Long et al. (2022) estudiaron
     GameStop usando solo títulos de WSB y funcionó. Además, WSB es el 55-60% del dataset,
     así que no podemos descartar los posts sin cuerpo.
   - En el resto de subreddits (r/stocks, r/investing, etc.) los posts suelen incluir
     análisis detallados en el cuerpo, así que concatenamos título + selftext.

Archivos que genera
-------------------
- data/clean/reddit_clean_daily.csv  → una fila por (fecha, ticker) con métricas diarias
- data/clean/reddit_clean_posts.csv  → posts individuales con texto_completo
- docs/figuras/reddit_nulos_antes.png
- docs/figuras/reddit_nulos_despues.png
- docs/figuras/reddit_posts_subreddit.png
- docs/figuras/reddit_posts_temporal.png
- docs/figuras/reddit_wordcloud_{TICKER}.png  (7 figuras, una por empresa)

Para ejecutar
-------------
    source venv/bin/activate
    python scripts/clean_reddit.py
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

# ── Configuración general ────────────────────────────────────────────────────
# Rutas, tickers, colores por empresa y por subreddit.

DIR_REDDIT     = os.path.join("data", "raw", "reddit")
RUTA_DAILY     = os.path.join("data", "clean", "reddit_clean_daily.csv")
RUTA_POSTS     = os.path.join("data", "clean", "reddit_clean_posts.csv")
DIR_FIGS       = os.path.join("docs", "figuras")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
COLORES = {
    "AAPL": "#555555", "MSFT": "#0078D4", "GOOGL": "#EA4335",
    "AMZN": "#FF9900", "NVDA": "#76B900", "META": "#1877F2", "TSLA": "#CC0000",
}
COLORES_SUB = {
    "wallstreetbets": "#FF4500",
    "stocks":         "#0DD3BB",
    "investing":      "#5F99CF",
    "StockMarket":    "#FFB000",
    "options":        "#9B59B6",
}

# Lista de bots que hemos identificado revisando manualmente los posts más
# frecuentes. Estos publican contenido automático (resúmenes pre-market,
# señales de trading, listas de seguimiento...) y no reflejan sentimiento
# real de inversores humanos.
BOTS = {
    "AutoModerator",      # el bot de moderación de Reddit, obvio
    "bigbear0083",        # publica resúmenes pre-market todos los días para todas las acciones
    "psychotrader00",     # genera señales de trading de forma automática
    "WinningWatchlist",   # crea listas de seguimiento automatizadas
    "WSBConsensus",       # agrega el "consenso" de WSB automáticamente
    "MillennialBets",     # publica resúmenes de las "apuestas" del día
    "callsonreddit",      # registra opciones call automáticamente
    "spxspyqqq666",       # alerta sobre movimientos de índices
    "intraalpha",         # señales algorítmicas automáticas
    "swaggymedia",        # contenido financiero automatizado
}

# Longitud mínima del texto para considerar un post útil para NLP
MIN_CHARS_TEXTO_COMPLETO = 10

# Palabras que excluimos de los word clouds: artefactos HTML, términos genéricos
# de Reddit, los propios nombres de las empresas y tickers (que obviamente
# aparecerían en todas partes) y términos financieros demasiado comunes.
STOPWORDS_EXTRA = {
    # Artefactos HTML/Markdown que aparecen en los posts de Reddit
    "nbsp", "amp", "gt", "lt", "x200b", "http", "https", "www", "com",
    "wallstreetbet", "wallstreetbets", "wsb",
    # Palabras super comunes en Reddit que no aportan nada al análisis
    "reddit", "post", "comment", "comments", "will", "like", "just",
    "think", "know", "get", "going", "would", "could", "should", "also",
    "one", "even", "still", "much", "really", "actually", "already",
    "need", "make", "good", "want", "say", "said", "see", "new", "well",
    "back", "time", "long", "way", "people", "edit", "update", "deleted",
    "removed", "nan", "re", "don", "ve", "ll", "isn", "didn", "doesn",
    "week", "month", "year", "day", "today", "now", "last", "next",
    "look", "use", "put", "call", "buy", "sell", "hold",  # demasiado genéricas
    # Los propios tickers y nombres de empresas (obvio que aparecen, no aportan)
    "aapl", "msft", "googl", "amzn", "nvda", "meta", "tsla",
    "apple", "microsoft", "alphabet", "amazon", "nvidia", "tesla",
    # Jerga financiera tan genérica que sale en todos los word clouds
    "stock", "stocks", "share", "shares", "market", "price", "trading",
    "option", "options", "company", "companies", "revenue", "earning",
    "earnings", "quarter", "billion", "million",
}

os.makedirs(DIR_FIGS, exist_ok=True)
os.makedirs("data/clean", exist_ok=True)


# ── Funciones auxiliares ──────────────────────────────────────────────────────
# Separadores para la terminal, guardado de figuras y limpieza de texto.

def separador(titulo: str):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print('='*60)


def guardar_figura(nombre: str):
    ruta = os.path.join(DIR_FIGS, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figura guardada: {ruta}")


def limpiar_texto(texto: str) -> str:
    """Limpia un texto de Reddit: quita las URLs, elimina caracteres raros
    (pero conserva cashtags como $AAPL y signos financieros como % y +)
    y normaliza los espacios en blanco."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+|www\S+", "", texto)       # fuera las URLs
    texto = re.sub(r"[^\w\s\$\%\+\-\.]", " ", texto)  # dejamos letras, números, $, %, +, -, .
    texto = re.sub(r"\s+", " ", texto).strip()          # compactamos espacios
    return texto


# ── 1. Carga de los CSVs ──────────────────────────────────────────────────────
# Leemos los 7 CSVs (uno por ticker) y los juntamos en un solo DataFrame.
# Normalizamos tipos porque algunos campos vienen como string o con nulos.

separador("1. CARGA DE DATOS")

dfs = []
for t in TICKERS:
    ruta_26 = os.path.join(DIR_REDDIT, f"{t}_reddit_2019_2026.csv")
    ruta = ruta_26 if os.path.exists(ruta_26) else os.path.join(DIR_REDDIT, f"{t}_reddit_2019_2025.csv")
    df_t = pd.read_csv(ruta, engine="python", on_bad_lines="skip")
    dfs.append(df_t)

df_raw = pd.concat(dfs, ignore_index=True)

# Forzamos los tipos correctos para cada columna. Algunos campos numéricos
# vienen como string por culpa de valores raros en los CSVs.
df_raw["score"]           = pd.to_numeric(df_raw["score"],           errors="coerce")
df_raw["num_comentarios"] = pd.to_numeric(df_raw["num_comentarios"], errors="coerce")
df_raw["upvote_ratio"]    = pd.to_numeric(df_raw["upvote_ratio"],    errors="coerce")
df_raw["created_utc"]     = pd.to_numeric(df_raw["created_utc"],     errors="coerce")
df_raw["fecha"]           = pd.to_datetime(df_raw["fecha"],          errors="coerce")
df_raw["titulo"]          = df_raw["titulo"].fillna("").astype(str)
df_raw["texto"]           = df_raw["texto"].fillna("").astype(str)
df_raw["autor"]           = df_raw["autor"].fillna("[desconocido]").astype(str)

n_raw = len(df_raw)
print(f"  Filas cargadas (7 CSVs combinados) : {n_raw:,}")
print(f"  Columnas                           : {df_raw.columns.tolist()}")
print(f"  Rango fechas                       : {df_raw['fecha'].min().date()} → {df_raw['fecha'].max().date()}")
print(f"\n  Posts por ticker:")
print(df_raw.groupby("ticker").size().to_string())


# ── 2. Revisión inicial (ANTES de tocar nada) ────────────────────────────────
# Miramos nulos, duplicados, posts borrados, bots y generamos la primera figura.

separador("2. REVISIÓN INICIAL — ANTES DE LIMPIEZA")

nulos_antes = df_raw.isnull().sum()
print(f"\n  Nulos por columna:")
print(nulos_antes[nulos_antes > 0].to_string())

print(f"\n  Duplicados por (id, ticker)        : {df_raw.duplicated(['id','ticker']).sum():,}")
print(f"  Posts con texto [removed]          : {(df_raw['texto'] == '[removed]').sum():,}")
print(f"  Posts con texto [deleted]          : {(df_raw['texto'] == '[deleted]').sum():,}")
print(f"  Posts con texto vacío              : {(df_raw['texto'].str.strip() == '').sum():,}")
print(f"  Posts de bots identificados        : {df_raw['autor'].isin(BOTS).sum():,}")

# Figura ANTES: cuántos posts tiene cada empresa, desglosado por subreddit
fig, ax = plt.subplots(figsize=(12, 5))
pivot = df_raw.groupby(["ticker","subreddit"]).size().unstack(fill_value=0)
pivot = pivot.reindex(columns=["wallstreetbets","stocks","investing","StockMarket","options"],
                      fill_value=0)
bottom = np.zeros(len(pivot))
for sub in pivot.columns:
    ax.bar(pivot.index, pivot[sub], bottom=bottom,
           label=sub, color=COLORES_SUB.get(sub, "#999"), alpha=0.85, edgecolor="white")
    bottom += pivot[sub].values
ax.set_title("Posts de Reddit por empresa y subreddit — ANTES de limpieza", fontsize=11)
ax.set_ylabel("Número de posts")
ax.legend(loc="upper right", fontsize=8)
for i, (t, total) in enumerate(pivot.sum(axis=1).items()):
    ax.text(i, total + 100, f"{total:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
guardar_figura("reddit_nulos_antes.png")


# ── 3. Limpieza ───────────────────────────────────────────────────────────────
# Aquí es donde aplicamos todas las decisiones de limpieza que describimos
# en el docstring: fechas, duplicados, bots, texto y longitud mínima.

separador("3. LIMPIEZA Y TRANSFORMACIONES")

df = df_raw.copy()

# 3a. Si la fecha no se puede parsear, no podemos situar el post en el tiempo
n0 = len(df)
df = df.dropna(subset=["fecha"])
print(f"  Fechas inválidas eliminadas        : {n0 - len(df):,}")

# 3b. Quitamos duplicados por (id, ticker). Si el mismo post aparece dos veces
# con el mismo ticker, es un duplicado real. Pero si aparece con tickers
# distintos es legítimo: un post que habla de $AAPL y $TSLA sale en ambos.
n0 = len(df)
df = df.drop_duplicates(subset=["id", "ticker"])
print(f"  Duplicados (id+ticker) eliminados  : {n0 - len(df):,}")

# 3c. Fuera los bots. Publican contenido repetitivo y automático (resúmenes
# diarios, alertas...) que no refleja sentimiento real de personas.
n0 = len(df)
df = df[~df["autor"].isin(BOTS)]
print(f"  Posts de bots eliminados           : {n0 - len(df):,}  (bots: {sorted(BOTS)})")

# 3d. Cuando el texto dice [removed] o [deleted] lo dejamos vacío, pero NO
# eliminamos el post. El título sigue ahí y muchas veces tiene toda la info
# que necesitamos (sobre todo en WSB).
df["texto"] = df["texto"].replace({"[removed]": "", "[deleted]": ""})

# 3e. Construimos texto_completo de forma diferente según el subreddit.
# En WSB usamos solo el título porque ahí es donde está el sentimiento
# (Long et al. 2022 lo demostraron con GameStop). En los demás subreddits
# concatenamos título + cuerpo porque los análisis suelen estar en el selftext.
wsb_mask = df["subreddit"] == "wallstreetbets"
df["texto_completo"] = ""
df.loc[wsb_mask, "texto_completo"] = df.loc[wsb_mask, "titulo"].apply(limpiar_texto)
df.loc[~wsb_mask, "texto_completo"] = (
    df.loc[~wsb_mask, "titulo"] + " " + df.loc[~wsb_mask, "texto"]
).str.strip().apply(limpiar_texto)

# 3f. Quitamos posts demasiado cortos (menos de 10 caracteres). Si el
# texto_completo es tan corto, no hay nada que analizar semánticamente.
# Excepción: los de WSB se quedan aunque sean cortísimos, porque en WSB
# incluso la cantidad de menciones diarias ya es una señal de sentimiento.
n0 = len(df)
no_wsb_cortos = (~wsb_mask) & (df["texto_completo"].str.len() < MIN_CHARS_TEXTO_COMPLETO)
df = df[~no_wsb_cortos]
print(f"  Posts con texto < {MIN_CHARS_TEXTO_COMPLETO} chars eliminados: {n0 - len(df):,}  (WSB exento)")

# 3g. Nos quedamos solo con posts dentro del periodo del estudio
n0 = len(df)
df = df[(df["fecha"] >= "2019-01-01") & (df["fecha"] <= "2026-04-02")]
print(f"  Posts fuera del periodo            : {n0 - len(df):,}")

n_limpio = len(df)
print(f"\n  Posts limpios totales              : {n_limpio:,} ({n_limpio/n_raw*100:.1f}%)")
print(f"\n  Posts por ticker tras limpieza:")
print(df.groupby("ticker").size().to_string())


# ── 4. Figuras ────────────────────────────────────────────────────────────────
# Generamos las visualizaciones: barras después de limpieza, evolución temporal
# y word clouds por empresa.

separador("4. FIGURAS")

# 4a. Misma figura que antes pero con datos limpios, para comparar
fig, ax = plt.subplots(figsize=(12, 5))
pivot_c = df.groupby(["ticker","subreddit"]).size().unstack(fill_value=0)
pivot_c = pivot_c.reindex(columns=["wallstreetbets","stocks","investing","StockMarket","options"],
                          fill_value=0)
bottom = np.zeros(len(pivot_c))
for sub in pivot_c.columns:
    ax.bar(pivot_c.index, pivot_c[sub], bottom=bottom,
           label=sub, color=COLORES_SUB.get(sub, "#999"), alpha=0.85, edgecolor="white")
    bottom += pivot_c[sub].values
ax.set_title("Posts de Reddit por empresa y subreddit — DESPUÉS de limpieza\n"
             "(bots eliminados, posts sin texto descartados)", fontsize=11)
ax.set_ylabel("Número de posts")
ax.legend(loc="upper right", fontsize=8)
for i, (t, total) in enumerate(pivot_c.sum(axis=1).items()):
    ax.text(i, total + 50, f"{total:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
guardar_figura("reddit_nulos_despues.png")

# 4b. Evolución mensual del número de posts por empresa. Se ven los picos
# de actividad en momentos como GameStop (ene 2021) o el boom de la IA.
fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
axes_flat = axes.flatten()
df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()

for i, ticker in enumerate(TICKERS):
    ax = axes_flat[i]
    sub_df = df[df["ticker"] == ticker]
    mensual = sub_df.groupby("mes").size().reset_index(name="n_posts")
    ax.fill_between(mensual["mes"], mensual["n_posts"],
                    color=COLORES[ticker], alpha=0.4)
    ax.plot(mensual["mes"], mensual["n_posts"], color=COLORES[ticker], linewidth=1.2)
    for fecha_str, label in [("2020-03-01","COVID"),("2021-01-01","GameStop"),
                               ("2022-02-01","META -30%"),("2023-05-01","Boom IA")]:
        ax.axvline(pd.Timestamp(fecha_str), color="#888", linestyle=":", linewidth=0.8)
    ax.set_title(ticker, fontsize=11, fontweight="bold", color=COLORES[ticker])
    ax.set_ylabel("Posts/mes", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

axes_flat[-1].set_visible(False)
fig.suptitle("Evolución mensual de posts Reddit — Magnificent 7 (2019–2026)",
             fontsize=12, y=1.01)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
guardar_figura("reddit_posts_temporal.png")

# 4c. Word clouds por empresa: las palabras más frecuentes en los posts de
# cada ticker. Generamos tanto un panel conjunto como figuras individuales.
print("\n  Generando word clouds...")
stopwords_wc = STOPWORDS.union(STOPWORDS_EXTRA)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes_flat = axes.flatten()

for i, ticker in enumerate(TICKERS):
    texto_ticker = " ".join(
        df[df["ticker"] == ticker]["texto_completo"].dropna().astype(str)
    )
    if not texto_ticker.strip():
        axes_flat[i].set_visible(False)
        continue

    wc = WordCloud(
        width=600, height=400,
        background_color="white",
        stopwords=stopwords_wc,
        colormap="RdYlGn",
        max_words=80,
        min_word_length=3,
        collocations=False,
    ).generate(texto_ticker)

    axes_flat[i].imshow(wc, interpolation="bilinear")
    axes_flat[i].axis("off")
    axes_flat[i].set_title(ticker, fontsize=13, fontweight="bold",
                            color=COLORES[ticker], pad=8)

    # Guardamos también cada word cloud por separado para el documento Word
    fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(f"Word Cloud Reddit — {ticker} (2019–2026)\n"
                    f"Términos más frecuentes en posts sobre {ticker}",
                    fontsize=12)
    plt.tight_layout()
    guardar_figura(f"reddit_wordcloud_{ticker}.png")

axes_flat[-1].set_visible(False)
fig.suptitle("Word Clouds Reddit — Magnificent 7 (2019–2026)\n"
             "Términos más frecuentes en posts limpios", fontsize=13, y=1.01)
plt.tight_layout()
guardar_figura("reddit_wordclouds_todos.png")


# ── 5. Agregación diaria ──────────────────────────────────────────────────────
# Agrupamos por fecha y ticker para generar métricas diarias: número de posts,
# score medio, mediana, comentarios, ratio de upvotes, porcentaje de WSB, etc.

separador("5. AGREGACIÓN DIARIA")

df["fecha_solo"] = df["fecha"].dt.date

diario = df.groupby(["fecha_solo", "ticker"]).agg(
    n_posts          = ("id",              "count"),
    score_mean       = ("score",           "mean"),
    score_std        = ("score",           "std"),
    score_median     = ("score",           "median"),
    n_comments_mean  = ("num_comentarios", "mean"),
    upvote_ratio_mean= ("upvote_ratio",    "mean"),
    pct_posts_wsb    = ("subreddit",       lambda x: (x == "wallstreetbets").mean() * 100),
    pct_con_texto    = ("texto",           lambda x: (x.str.len() > 0).mean() * 100),
).reset_index()

diario.rename(columns={"fecha_solo": "fecha"}, inplace=True)
diario["fecha"] = pd.to_datetime(diario["fecha"])
diario = diario.sort_values(["ticker", "fecha"]).reset_index(drop=True)

print(f"  Filas en agregación diaria         : {len(diario):,}")
print(f"  Rango fechas                       : {diario['fecha'].min().date()} → {diario['fecha'].max().date()}")
print(f"\n  Días únicos con posts por ticker:")
for t in TICKERS:
    sub = diario[diario["ticker"] == t]
    print(f"    {t}: {len(sub):>4,} días | "
          f"posts/día media: {sub['n_posts'].mean():.1f} | "
          f"score medio: {sub['score_mean'].mean():.1f}")


# ── 6. Guardar los resultados ─────────────────────────────────────────────────
# Exportamos los posts individuales (para FinBERT) y la agregación diaria
# (para los modelos).

separador("6. GUARDAR ARCHIVOS")

# Posts individuales ya limpios, listos para pasar por FinBERT en la Fase 3
cols_posts = ["id", "fecha", "ticker", "subreddit", "titulo", "texto",
              "texto_completo", "autor", "score", "num_comentarios",
              "upvote_ratio", "url"]
df[cols_posts].to_csv(RUTA_POSTS, index=False, encoding="utf-8")
print(f"  Guardado: {RUTA_POSTS}  ({len(df):,} posts)")

# Agregación diaria para los modelos
cols_daily = ["fecha", "ticker", "n_posts", "score_mean", "score_std",
              "score_median", "n_comments_mean", "upvote_ratio_mean",
              "pct_posts_wsb", "pct_con_texto"]
diario[cols_daily].to_csv(RUTA_DAILY, index=False, encoding="utf-8")
print(f"  Guardado: {RUTA_DAILY}  ({len(diario):,} filas)")


# ── 7. Resumen final ──────────────────────────────────────────────────────────
# Imprimimos todo: cuántos posts había, cuántos quitamos y por qué, estadísticas
# por ticker y la lista de figuras generadas.

separador("7. RESUMEN DE LA LIMPIEZA")

eliminados = n_raw - n_limpio
print(f"""
  DATOS DE ENTRADA
  ─────────────────────────────────────────────
  Posts en crudo (7 CSVs)      : {n_raw:>8,}
  Periodo                      : 2019-01-01 → 2025-12-31
  Subreddits                   : wallstreetbets, stocks, investing, StockMarket, options

  ELIMINADOS
  ─────────────────────────────────────────────
  Fechas inválidas             : incluidos arriba
  Duplicados (id+ticker)       : {df_raw.duplicated(['id','ticker']).sum():>8,}
  Posts de bots                : incluidos arriba
  Texto < {MIN_CHARS_TEXTO_COMPLETO} caracteres          : incluidos arriba
  ─────────────────────────────────────────────
  TOTAL eliminados             : {eliminados:>8,} ({eliminados/n_raw*100:.1f}%)

  DATOS DE SALIDA
  ─────────────────────────────────────────────
  Posts limpios (posts)        : {n_limpio:>8,} ({n_limpio/n_raw*100:.1f}%)
  Filas agregación diaria      : {len(diario):>8,}

  ESTADÍSTICAS (posts limpios)
  ─────────────────────────────────────────────""")

for t in TICKERS:
    sub = df[df["ticker"] == t]
    print(f"  {t}: {len(sub):>6,} posts | "
          f"score medio: {sub['score'].mean():.1f} | "
          f"con selftext: {(sub['texto'].str.len()>0).mean()*100:.0f}%")

print(f"""
  FIGURAS GENERADAS
  ─────────────────────────────────────────────
  docs/figuras/reddit_nulos_antes.png
  docs/figuras/reddit_nulos_despues.png
  docs/figuras/reddit_posts_temporal.png
  docs/figuras/reddit_wordclouds_todos.png
  docs/figuras/reddit_wordcloud_{{TICKER}}.png  (7 figuras)
""")
