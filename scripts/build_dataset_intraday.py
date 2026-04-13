"""
Script: build_dataset_intraday.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Este script construye el dataset integrado para el análisis intradía (horario).
Junta tres cosas:
  - Los datos de mercado horarios que descargamos (barras de 1 hora)
  - El sentimiento GDELT diario, filtrado al periodo abr 2024 – abr 2026
  - La actividad de Reddit diaria, mismo periodo

El truco es que el sentimiento es diario pero las barras son horarias, así que
cada barra de un mismo día hereda las mismas métricas de sentimiento. El merge
se hace por (date, ticker) con left join desde las barras, igual que en el
dataset diario.

Lo hacemos como dataset separado porque el dataset_integrado.csv original cubre
2019–2025 en escala diaria y está pensado para GARCH y GARCH-X. Este nuevo
dataset está diseñado específicamente para:
  - Los modelos XGBoost horarios con sentimiento como variables
  - Análisis de cómo el sentimiento afecta a la volatilidad hora a hora
  - FinBERT aplicado al periodo intradía

Archivo que genera
------------------
  - data/clean/dataset_intraday_integrado.csv

Para ejecutar
-------------
    source venv/bin/activate
    python scripts/build_dataset_intraday.py
"""

import pandas as pd
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
# Rutas a los archivos de entrada y salida, periodo temporal y tickers.

BASE = Path(__file__).resolve().parent.parent

RUTA_INTRADAY = BASE / "data" / "raw" / "market_intraday" / "ALL_market_intraday_2024_2026.csv"
RUTA_GDELT    = BASE / "data" / "clean" / "news_clean_daily.csv"
RUTA_REDDIT   = BASE / "data" / "clean" / "reddit_clean_daily.csv"
RUTA_SALIDA   = BASE / "data" / "clean" / "dataset_intraday_integrado.csv"

FECHA_MIN = "2024-04-01"
FECHA_MAX = "2026-04-02"

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


# ── Carga y merge ────────────────────────────────────────────────────────────
# Todo el proceso va en main(): cargamos las tres fuentes, las juntamos y
# exportamos el resultado.

def main():
    print("=" * 60)
    print("  Construcción dataset intradía integrado")
    print("=" * 60)

    # 1. Cargamos las barras horarias y nos quedamos solo con la sesión regular
    print("\n  Cargando datos intradía...")
    df = pd.read_csv(RUTA_INTRADAY, parse_dates=["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    # Solo nos quedamos con las barras de la sesión regular (9:30 a 16:00)
    df = df[df["es_sesion_regular"] == True].copy()
    print(f"  Barras horarias: {len(df):,} ({df.date.min().date()} → {df.date.max().date()})")

    # 2. Cargamos el sentimiento GDELT diario y lo filtramos al periodo intradía.
    # Renombramos las columnas con prefijo gdelt_ para evitar colisiones.
    print("\n  Cargando GDELT diario...")
    gdelt = pd.read_csv(RUTA_GDELT, parse_dates=["fecha"])
    gdelt = gdelt[(gdelt.fecha >= FECHA_MIN) & (gdelt.fecha <= FECHA_MAX)]
    gdelt = gdelt.rename(columns={
        "fecha": "date",
        "tone_mean": "gdelt_tone_mean",
        "tone_std": "gdelt_tone_std",
        "tone_median": "gdelt_tone_median",
        "tone_min": "gdelt_tone_min",
        "tone_max": "gdelt_tone_max",
        "n_noticias": "gdelt_n_noticias",
        "pct_negativo": "gdelt_pct_negativo",
        "pct_positivo": "gdelt_pct_positivo",
    })
    print(f"  GDELT: {len(gdelt):,} filas ({gdelt.date.min().date()} → {gdelt.date.max().date()})")

    # 3. Lo mismo con Reddit: cargamos, filtramos y renombramos con prefijo reddit_
    print("\n  Cargando Reddit diario...")
    reddit = pd.read_csv(RUTA_REDDIT, parse_dates=["fecha"])
    reddit = reddit[(reddit.fecha >= FECHA_MIN) & (reddit.fecha <= FECHA_MAX)]
    reddit = reddit.rename(columns={
        "fecha": "date",
        "n_posts": "reddit_n_posts",
        "score_mean": "reddit_score_mean",
        "score_std": "reddit_score_std",
        "score_median": "reddit_score_median",
        "n_comments_mean": "reddit_n_comments_mean",
        "upvote_ratio_mean": "reddit_upvote_ratio_mean",
        "pct_posts_wsb": "reddit_pct_posts_wsb",
        "pct_con_texto": "reddit_pct_con_texto",
    })
    print(f"  Reddit: {len(reddit):,} filas ({reddit.date.min().date()} → {reddit.date.max().date()})")

    # 4. Hacemos left join: las barras horarias se quedan con el sentimiento
    # del día correspondiente. Cada barra del mismo día tendrá los mismos
    # valores de sentimiento (porque GDELT y Reddit son diarios).
    print("\n  Merging...")
    cols_gdelt = [c for c in gdelt.columns if c.startswith("gdelt_")] + ["date", "ticker"]
    cols_reddit = [c for c in reddit.columns if c.startswith("reddit_")] + ["date", "ticker"]

    df = df.merge(gdelt[cols_gdelt], on=["date", "ticker"], how="left")
    df = df.merge(reddit[cols_reddit], on=["date", "ticker"], how="left")

    # 5. Revisamos qué porcentaje de barras tiene cobertura de sentimiento
    n_total = len(df)
    n_gdelt = df["gdelt_tone_mean"].notna().sum()
    n_reddit = df["reddit_n_posts"].notna().sum()

    print(f"\n  Resultado:")
    print(f"  Barras totales:              {n_total:,}")
    print(f"  Con GDELT (tone):            {n_gdelt:,} ({100*n_gdelt/n_total:.1f}%)")
    print(f"  Con Reddit (posts):          {n_reddit:,} ({100*n_reddit/n_total:.1f}%)")
    print(f"  Columnas:                    {len(df.columns)}")

    # Desglose de cobertura por empresa
    print(f"\n  Cobertura por ticker:")
    for t in TICKERS:
        sub = df[df.ticker == t]
        g = sub.gdelt_tone_mean.notna().sum()
        r = sub.reddit_n_posts.notna().sum()
        print(f"    {t}: {len(sub):,} barras | GDELT {100*g/len(sub):.0f}% | Reddit {100*r/len(sub):.0f}%")

    # 6. Exportamos el dataset final
    df.to_csv(RUTA_SALIDA, index=False)
    print(f"\n  Guardado: {RUTA_SALIDA}")
    print(f"  Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()
