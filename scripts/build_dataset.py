"""
build_dataset.py — Integración de las tres fuentes limpias en un dataset unico

Este script es el que junta todo: combina los datos limpios de mercado, noticias GDELT
y Reddit en un unico DataFrame donde cada fila es un dia habil para una empresa.

La lógica es sencilla: el mercado manda. El calendario de referencia son los días
hábiles bursátiles (market_clean.csv), y las fuentes de sentimiento se enganchan con
left joins por (date, ticker). Si un día hay noticias o posts pero no hay mercado
(fin de semana, festivo), esos registros se caen solos. Si un día hay mercado pero
no hay noticias ni posts, se conserva con n_noticias=0 y n_posts=0, porque la
ausencia de cobertura mediática ya es información relevante en sí misma
(Baker & Wurgler, 2006).

Archivo que genera
------------------
data/clean/dataset_integrado.csv — el dataset final con mercado + GDELT + Reddit,
todo alineado en el calendario bursátil y listo para los modelos.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# RUTAS a los datasets limpios de cada fuente
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

PATH_MARKET = CLEAN_DIR / "market_clean.csv"
PATH_NEWS   = CLEAN_DIR / "news_clean_daily.csv"
PATH_REDDIT = CLEAN_DIR / "reddit_clean_daily.csv"
PATH_OUTPUT = CLEAN_DIR / "dataset_integrado.csv"


def cargar_datasets():
    """Carga los tres CSVs limpios (mercado, noticias, Reddit) y renombra las
    columnas de fecha para que todas se llamen 'date'. Así luego el merge es directo."""
    print("── Cargando datasets limpios ──")

    market = pd.read_csv(PATH_MARKET, parse_dates=["date"])
    news   = pd.read_csv(PATH_NEWS,   parse_dates=["fecha"])
    reddit = pd.read_csv(PATH_REDDIT, parse_dates=["fecha"])

    # News y Reddit usan 'fecha', mercado usa 'date'. Las renombramos todas a 'date'.
    news   = news.rename(columns={"fecha": "date"})
    reddit = reddit.rename(columns={"fecha": "date"})

    print(f"  Mercado : {market.shape[0]:>7,} filas × {market.shape[1]} columnas")
    print(f"  Noticias: {news.shape[0]:>7,} filas × {news.shape[1]} columnas")
    print(f"  Reddit  : {reddit.shape[0]:>7,} filas × {reddit.shape[1]} columnas")

    return market, news, reddit


def prefijar_columnas(df, prefijo, excluir=("date", "ticker")):
    """Pone un prefijo (como 'news_' o 'rddt_') a todas las columnas excepto
    las que usamos para el join (date y ticker), para evitar que se pisen
    nombres entre fuentes."""
    return df.rename(columns={
        c: f"{prefijo}_{c}" for c in df.columns if c not in excluir
    })


def hacer_joins(market, news, reddit):
    """
    Hace left joins de noticias y Reddit sobre el mercado usando (date, ticker).
    Al ser left join desde mercado, nos aseguramos de que el resultado tiene
    exactamente los mismos días hábiles que el mercado, ni más ni menos.
    Si el join cambia el número de filas, algo va mal (hay duplicados en alguna
    fuente de sentimiento) y lanzamos un error.
    """
    print("\n── Joins ──")

    # Prefijamos las columnas para que no se pisen (news_tone_mean vs rddt_score_mean, etc.)
    news_pref   = prefijar_columnas(news,   "news")
    reddit_pref = prefijar_columnas(reddit, "rddt")

    n_base = len(market)

    df = market.merge(news_pref,   on=["date", "ticker"], how="left")
    n_tras_news = len(df)

    df = df.merge(reddit_pref, on=["date", "ticker"], how="left")
    n_tras_reddit = len(df)

    print(f"  Base (mercado)          : {n_base:>7,} filas")
    print(f"  Tras join noticias      : {n_tras_news:>7,} filas  (delta: {n_tras_news - n_base:+d})")
    print(f"  Tras join Reddit        : {n_tras_reddit:>7,} filas  (delta: {n_tras_reddit - n_base:+d})")

    if n_tras_reddit != n_base:
        raise ValueError(
            f"El join alteró el número de filas ({n_base} → {n_tras_reddit}). "
            "Hay duplicados en las claves de alguna fuente de sentimiento."
        )

    return df


def tratar_nulos(df):
    """
    No todos los nulos son iguales, así que los tratamos de forma diferente
    según de dónde vengan:

    1. Las volatilidades rolling de mercado (vol_5d, vol_20d) tienen NaN al
       principio de cada ticker porque la ventana no tiene datos suficientes.
       Los rellenamos con forward-fill: básicamente cogemos el primer valor
       disponible y lo propagamos hacia atrás dentro de cada ticker.

    2. Las variables de noticias GDELT: los días sin noticias son huecos reales.
       Los conteos (n_noticias, pct_negativo, pct_positivo) los ponemos a 0
       porque "0 noticias" es un dato, no un nulo. Pero el tono medio y demás
       métricas se dejan como NaN porque no tiene sentido inventar un tono
       cuando no hubo cobertura ese día.

    3. Las variables de Reddit: misma lógica. Los conteos a 0, las métricas
       como NaN. No podemos calcular un score medio si no hubo posts.
    """
    print("\n── Tratamiento de nulos ──")

    # ── 1. Mercado: rellenamos con forward-fill por ticker ──
    cols_ffill = ["vol_realized_5d", "vol_realized_20d"]
    nulos_antes = {c: df[c].isna().sum() for c in cols_ffill}

    df = df.sort_values(["ticker", "date"])
    df[cols_ffill] = df.groupby("ticker")[cols_ffill].transform(
        lambda s: s.ffill()
    )

    nulos_despues = {c: df[c].isna().sum() for c in cols_ffill}
    for c in cols_ffill:
        print(f"  {c}: {nulos_antes[c]} NaN → {nulos_despues[c]} NaN  (ffill por ticker)")

    # ── 2. Noticias: ponemos ceros en los conteos, dejamos NaN en las métricas ──
    cols_news_cero = ["news_n_noticias", "news_pct_negativo", "news_pct_positivo"]
    for c in cols_news_cero:
        n = df[c].isna().sum()
        df[c] = df[c].fillna(0.0)
        print(f"  {c}: {n} NaN → 0.0  (días sin noticias)")

    cols_news_nan = ["news_tone_mean", "news_tone_std", "news_tone_median",
                     "news_tone_min",  "news_tone_max"]
    for c in cols_news_nan:
        n = df[c].isna().sum()
        print(f"  {c}: {n} NaN conservados  (tono no imputable sin noticias)")

    # ── 3. Reddit: mismo criterio que noticias ──
    cols_rddt_cero = ["rddt_n_posts", "rddt_pct_posts_wsb", "rddt_pct_con_texto"]
    for c in cols_rddt_cero:
        n = df[c].isna().sum()
        df[c] = df[c].fillna(0.0)
        print(f"  {c}: {n} NaN → 0.0  (días sin posts)")

    cols_rddt_nan = ["rddt_score_mean", "rddt_score_std", "rddt_score_median",
                     "rddt_n_comments_mean", "rddt_upvote_ratio_mean"]
    for c in cols_rddt_nan:
        n = df[c].isna().sum()
        print(f"  {c}: {n} NaN conservados  (métricas no imputables sin posts)")

    return df


def añadir_calendario(df):
    """Añade columnas de calendario (día de la semana, mes, trimestre, año).
    Son útiles como features en XGBoost para capturar estacionalidad."""
    print("\n── Variables de calendario ──")
    df["day_of_week"] = df["date"].dt.dayofweek      # 0 = lunes
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["year"]        = df["date"].dt.year
    print("  Añadidas: day_of_week, month, quarter, year")
    return df


def validar(df):
    """Comprobaciones finales para asegurarnos de que todo está bien: no hay
    duplicados en la clave primaria, los tickers son los 7 esperados, y
    revisamos la cobertura de sentimiento por empresa."""
    print("\n── Validación final ──")

    # Dimensiones del dataset final
    n_filas, n_cols = df.shape
    print(f"  Shape: {n_filas:,} filas × {n_cols} columnas")

    # Verificamos que están las 7 empresas
    tickers = sorted(df["ticker"].unique())
    print(f"  Tickers ({len(tickers)}): {tickers}")

    # Cuántos días hábiles tiene cada empresa
    for t in tickers:
        n = (df["ticker"] == t).sum()
        print(f"    {t}: {n} días")

    # Periodo temporal cubierto
    print(f"  Fecha mín: {df['date'].min().date()}")
    print(f"  Fecha máx: {df['date'].max().date()}")

    # Nos aseguramos de que no haya duplicados en (date, ticker)
    n_dup = df.duplicated(subset=["date", "ticker"]).sum()
    print(f"  Duplicados (date+ticker): {n_dup}")
    if n_dup > 0:
        raise ValueError("Hay duplicados en la clave primaria date+ticker.")

    # Revisamos qué nulos quedan y de dónde vienen
    print("\n  Nulos por columna:")
    nulos = df.isna().sum()
    nulos_no_cero = nulos[nulos > 0]
    if nulos_no_cero.empty:
        print("    (ninguno)")
    else:
        for col, n in nulos_no_cero.items():
            pct = 100 * n / len(df)
            print(f"    {col}: {n:,} ({pct:.1f}%)")

    # Miramos qué porcentaje de días tiene cobertura de noticias y Reddit
    print("\n  Cobertura de sentimiento por ticker:")
    print(f"  {'Ticker':<8} {'Días merc.':<12} {'Con noticias':<14} {'Con Reddit':<12}")
    for t in tickers:
        sub = df[df["ticker"] == t]
        dias = len(sub)
        con_news   = (sub["news_n_noticias"] > 0).sum()
        con_reddit = (sub["rddt_n_posts"] > 0).sum()
        print(f"  {t:<8} {dias:<12} {con_news:<14} ({100*con_news/dias:.0f}%)  "
              f"{con_reddit:<12} ({100*con_reddit/dias:.0f}%)")


def guardar(df):
    """Exporta el dataset final a CSV, ordenado por ticker y fecha."""
    df_sorted = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_sorted.to_csv(PATH_OUTPUT, index=False)
    size_mb = PATH_OUTPUT.stat().st_size / 1e6
    print(f"\n── Guardado ──")
    print(f"  {PATH_OUTPUT}")
    print(f"  {df_sorted.shape[0]:,} filas × {df_sorted.shape[1]} columnas")
    print(f"  Tamaño en disco: {size_mb:.1f} MB")
    return df_sorted


def main():
    print("=" * 60)
    print("BUILD DATASET — Integración ETL final")
    print("=" * 60)

    market, news, reddit = cargar_datasets()
    df = hacer_joins(market, news, reddit)
    df = tratar_nulos(df)
    df = añadir_calendario(df)
    validar(df)
    df = guardar(df)

    print("\n✓ dataset_integrado.csv generado correctamente.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
