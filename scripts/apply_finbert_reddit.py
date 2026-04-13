"""
Script: apply_finbert_reddit.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

En resumen, ¿qué hace esto?
----------------------------
Este script coge todos los posts de Reddit que tenemos limpios y se los pasa uno a uno
(bueno, en batches) al modelo FinBERT (ProsusAI/finbert) para que clasifique el
sentimiento financiero de cada texto. FinBERT es un BERT que fue pre-entrenado
específicamente con textos del mundo financiero, así que sabe distinguir bastante bien si
un post es:
  - positive (alcista, optimista, "to the moon" y esas cosas)
  - negative (bajista, pesimista, "esto se hunde")
  - neutral (ni fu ni fa, no se moja)

Al final genera dos archivos de salida:
  1. Uno con cada post individual y su clasificación FinBERT (reddit_finbert_posts.csv)
  2. Otro con la agregación a nivel diario por ticker (reddit_finbert_daily.csv), que es
     el que luego usan los modelos predictivos

¿Y por qué FinBERT en vez de otros modelos de sentimiento?
-----------------------------------------------------------
La razón principal es que FinBERT (Araci, 2019) fue entrenado con el corpus Financial
PhraseBank y noticias de Reuters, así que entiende el lenguaje financiero de verdad.
Modelos genéricos como VADER o TextBlob no pillan que "bearish" es negativo o que
"bullish" es positivo en este contexto. En la literatura reciente de NLP aplicado a
finanzas, FinBERT es prácticamente el estándar, así que tiene todo el sentido usarlo aquí.

¿Qué archivos genera?
----------------------
- data/clean/reddit_finbert_posts.csv   → cada post con su label y score de FinBERT
- data/clean/reddit_finbert_daily.csv   → sentimiento agregado por día y ticker

Las columnas que se calculan a nivel diario son:
- finbert_positive_pct  : porcentaje de posts positivos ese día
- finbert_negative_pct  : porcentaje de posts negativos ese día
- finbert_neutral_pct   : porcentaje de posts neutros ese día
- finbert_sentiment_score: un score compuesto que va de -1 a +1
- finbert_n_posts       : cuántos posts se procesaron ese día (para saber si la muestra es fiable)

Para ejecutarlo
---------------
    source venv/bin/activate
    python scripts/apply_finbert_reddit.py
"""

import os
import sys
import time
import logging

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Configuración ────────────────────────────────────────────────────────────
# Aquí van los parámetros principales del script. Si algún día hay que tocar algo
# (cambiar el modelo, ajustar el tamaño de batch, etc.), es aquí donde se hace.

MODELO = "ProsusAI/finbert"
BATCH_SIZE = 64
MAX_LENGTH = 512  # FinBERT soporta hasta 512 tokens, así que le damos el post entero siempre que se pueda
MAX_CHARS  = 1500 # Si un post tiene más de 1500 caracteres (~250 palabras), lo cortamos antes de tokenizar para no pasarnos de RAM

RUTA_POSTS_IN  = os.path.join("data", "clean", "reddit_clean_posts.csv")
RUTA_POSTS_OUT = os.path.join("data", "clean", "reddit_finbert_posts.csv")
RUTA_DAILY_OUT = os.path.join("data", "clean", "reddit_finbert_daily.csv")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Funciones ────────────────────────────────────────────────────────────────

def cargar_modelo():
    """Descarga (si es la primera vez) y carga en memoria el tokenizer y el modelo FinBERT.
    La primera ejecución tarda un poco porque baja los pesos del modelo desde HuggingFace,
    pero luego quedan cacheados y va rápido."""
    log.info(f"Cargando modelo {MODELO}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO)
    model.eval()
    log.info("Modelo cargado correctamente")
    return tokenizer, model


def clasificar_batch(textos: list[str], tokenizer, model) -> list[dict]:
    """
    Le pasa un grupo de textos a FinBERT de golpe (en batch) y devuelve para cada uno
    su etiqueta de sentimiento (positive/negative/neutral) junto con las probabilidades
    de cada clase. Así no hay que hacer una inferencia por cada post, que sería lentísimo.
    """
    # Tokenizamos todos los textos del batch a la vez, con truncación por si alguno
    # se pasa de los 512 tokens, y padding dinámico (se rellena hasta el más largo del batch)
    inputs = tokenizer(
        textos,
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Pasamos los logits por softmax para convertirlos en probabilidades (que sumen 1)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # El orden de las clases en FinBERT es: índice 0 = positive, 1 = negative, 2 = neutral
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    resultados = []
    for i in range(len(textos)):
        pred_idx = probs[i].argmax().item()
        resultados.append({
            "finbert_label": label_map[pred_idx],
            "finbert_score": probs[i][pred_idx].item(),
            "finbert_positive": probs[i][0].item(),
            "finbert_negative": probs[i][1].item(),
            "finbert_neutral": probs[i][2].item(),
        })

    return resultados


def procesar_posts(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """Recorre todos los posts del DataFrame y los clasifica con FinBERT en batches.

    Hay un truco importante aquí: antes de procesar, ordenamos los posts por longitud
    de texto. La idea es que si agrupamos posts cortos con posts cortos, el padding
    necesario dentro de cada batch es mínimo y se procesan volando. Los posts largos
    (que son los menos) van en sus propios batches con batch size más pequeño para
    no reventar la memoria. Al terminar, restauramos el orden original para que
    todo cuadre con el DataFrame de entrada.
    """
    # Primero ordenamos por longitud de texto para que los batches sean más homogéneos
    df = df.copy()
    df["_texto_len"] = df["texto_completo"].fillna("").str.len()
    df["_orig_idx"] = range(len(df))
    df_sorted = df.sort_values("_texto_len").reset_index(drop=True)

    textos = df_sorted["texto_completo"].tolist()
    longitudes = df_sorted["_texto_len"].tolist()
    n_total = len(textos)

    log.info(f"Procesando {n_total:,} posts con batch size adaptativo")
    log.info(f"Posts ordenados por longitud — padding dinámico por batch")

    todos_resultados = []
    t_inicio = time.time()
    i = 0
    batch_count = 0

    while i < n_total:
        # El tamaño del batch se adapta según lo largo que sea el texto:
        # posts cortitos (<200 chars) van en batches de 64, los medianos de 32,
        # y los largos de 8 para no pasarnos de RAM
        largo_actual = longitudes[min(i + BATCH_SIZE - 1, n_total - 1)]
        if largo_actual < 200:
            bs = 64
        elif largo_actual < 600:
            bs = 32
        else:
            bs = 8  # Posts largos: mejor ir de pocos en pocos para no quedarnos sin memoria

        batch = textos[i:i + bs]

        # Si algún texto es absurdamente largo, lo cortamos antes de mandárselo al tokenizer
        batch_limpio = []
        for t in batch:
            if not isinstance(t, str) or len(t.strip()) == 0:
                batch_limpio.append("neutral")
            else:
                batch_limpio.append(t[:MAX_CHARS])

        resultados = clasificar_batch(batch_limpio, tokenizer, model)
        todos_resultados.extend(resultados)

        i += bs
        batch_count += 1

        # Cada 25 batches (o al final) mostramos cómo vamos: porcentaje, velocidad y tiempo estimado
        if batch_count % 25 == 0 or i >= n_total:
            elapsed = time.time() - t_inicio
            procesados = min(i, n_total)
            pct = 100 * procesados / n_total
            posts_s = procesados / elapsed if elapsed > 0 else 0
            restantes = n_total - procesados
            eta_min = (restantes / posts_s / 60) if posts_s > 0 else 0
            log.info(
                f"  {procesados:>6,}/{n_total:,} posts | "
                f"{pct:5.1f}% | "
                f"{posts_s:.0f} posts/s | "
                f"bs={bs} | "
                f"Quedan: {eta_min:.1f} min"
            )

    # Pegamos los resultados de FinBERT al DataFrame (que todavía está ordenado por longitud)
    for key in ["finbert_label", "finbert_score", "finbert_positive",
                "finbert_negative", "finbert_neutral"]:
        df_sorted[key] = [r[key] for r in todos_resultados]

    # Ahora sí, volvemos al orden original para que las filas cuadren con el CSV de entrada
    df_result = df_sorted.sort_values("_orig_idx").reset_index(drop=True)
    df_result = df_result.drop(columns=["_texto_len", "_orig_idx"])

    t_total = time.time() - t_inicio
    log.info(f"Procesamiento completado en {t_total/60:.1f} minutos ({t_total/n_total*1000:.0f} ms/post)")

    return df_result


def agregar_diario(df: pd.DataFrame) -> pd.DataFrame:
    """Toma los posts individuales ya clasificados y los agrega a nivel diario por ticker.
    Básicamente resume: de todos los posts de AAPL del día tal, cuántos fueron positivos,
    cuántos negativos, cuántos neutros, y cuál es el sentimiento medio del día."""

    def sentiment_score(grupo):
        """Calcula un score compuesto restando la probabilidad negativa de la positiva
        y promediando. El resultado va de -1 (todo negativo) a +1 (todo positivo)."""
        return (grupo["finbert_positive"] - grupo["finbert_negative"]).mean()

    diario = df.groupby(["fecha", "ticker"]).agg(
        finbert_positive_pct=("finbert_label", lambda x: (x == "positive").mean() * 100),
        finbert_negative_pct=("finbert_label", lambda x: (x == "negative").mean() * 100),
        finbert_neutral_pct=("finbert_label", lambda x: (x == "neutral").mean() * 100),
        finbert_sentiment_score=("finbert_positive", lambda x: (
            x.values - df.loc[x.index, "finbert_negative"].values
        ).mean()),
        finbert_n_posts=("finbert_label", "count"),
    ).reset_index()

    diario["fecha"] = pd.to_datetime(diario["fecha"])
    diario = diario.sort_values(["ticker", "fecha"]).reset_index(drop=True)

    return diario


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  FinBERT — Análisis de sentimiento Reddit")
    log.info("=" * 60)

    # 1. Cargamos los posts limpios de Reddit (ya filtrados y con texto)
    log.info("Cargando posts...")
    df = pd.read_csv(RUTA_POSTS_IN, parse_dates=["fecha"])

    # Nos quedamos solo con los que tienen texto de verdad (descartamos los vacíos o nulos)
    mask = df["texto_completo"].notna() & (df["texto_completo"].str.strip() != "")
    df = df[mask].copy()
    log.info(f"Posts con texto: {len(df):,}")

    # 2. Cargamos FinBERT en memoria (la primera vez descarga los pesos de HuggingFace)
    tokenizer, model = cargar_modelo()

    # 3. Le pasamos todos los posts a FinBERT (esto es lo que tarda, pero muestra progreso)
    df_result = procesar_posts(df, tokenizer, model)

    # 4. Guardamos el CSV con cada post y su clasificación de sentimiento
    df_result.to_csv(RUTA_POSTS_OUT, index=False, encoding="utf-8")
    log.info(f"Guardado: {RUTA_POSTS_OUT} ({len(df_result):,} posts)")

    # 5. Mostramos cómo quedó la distribución global de sentimiento (para hacernos una idea rápida)
    log.info("\nDistribución global de sentimiento:")
    for label in ["positive", "negative", "neutral"]:
        n = (df_result["finbert_label"] == label).sum()
        log.info(f"  {label:10s}: {n:>7,} ({100*n/len(df_result):.1f}%)")
    log.info(f"  Score medio: {(df_result['finbert_positive'] - df_result['finbert_negative']).mean():+.4f}")

    # 6. Lo mismo pero desglosado por empresa, para ver si alguna es más positiva/negativa
    log.info("\nPor ticker:")
    for t in TICKERS:
        sub = df_result[df_result["ticker"] == t]
        pos = (sub["finbert_label"] == "positive").mean() * 100
        neg = (sub["finbert_label"] == "negative").mean() * 100
        score = (sub["finbert_positive"] - sub["finbert_negative"]).mean()
        log.info(f"  {t}: pos={pos:.1f}% neg={neg:.1f}% score={score:+.4f}")

    # 7. Por último, agregamos todo a nivel diario (un registro por día-ticker) para los modelos
    log.info("\nAgregando a nivel diario...")
    diario = agregar_diario(df_result)
    diario.to_csv(RUTA_DAILY_OUT, index=False, encoding="utf-8")
    log.info(f"Guardado: {RUTA_DAILY_OUT} ({len(diario):,} filas)")
    log.info(f"Periodo: {diario['fecha'].min().date()} → {diario['fecha'].max().date()}")

    log.info("\n  FinBERT completado.")


if __name__ == "__main__":
    main()
