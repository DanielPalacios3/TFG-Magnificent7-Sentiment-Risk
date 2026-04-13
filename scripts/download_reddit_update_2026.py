"""
Script: download_reddit_update_2026.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Este script es un "parche" que descarga los posts de Reddit correspondientes a
enero-marzo de 2026, que son los 3 meses que nos faltaban para que la fuente de
texto de Reddit esté alineada con los datos intradía del mercado (que llegan hasta
abril 2026). Coge los posts nuevos y los añade a los CSVs existentes en
data/raw/reddit/ sin meter duplicados.

¿Y por qué un script separado en vez de usar download_reddit.py?
-----------------------------------------------------------------
El script original (download_reddit.py) descarga todo el periodo 2019-2025 de golpe
y tiene una lógica de skip por subreddit. Este en cambio es incremental: solo
descarga el trozo que falta (enero-marzo 2026) y lo appende a lo que ya teníamos.
Es más limpio y rápido que relanzar la descarga completa.

¿Qué archivos modifica?
-----------------------
Actualiza los CSVs existentes (y crea versiones con nombre actualizado):
- data/raw/reddit/{TICKER}_reddit_2019_2025.csv  (se actualiza y se copia a 2019_2026)
- data/raw/reddit/ALL_reddit_2019_2025.csv       (ídem)

Uso
---
    source venv/bin/activate
    python scripts/download_reddit_update_2026.py
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ── Configuración — solo el periodo que falta por descargar ──────────────────

TICKERS = {
    "AAPL": "$AAPL", "MSFT": "$MSFT", "GOOGL": "$GOOGL",
    "AMZN": "$AMZN", "NVDA": "$NVDA", "META": "$META", "TSLA": "$TSLA",
}

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]

# Solo descargamos el trozo que nos faltaba: de enero a principios de abril de 2026
FECHA_INICIO = datetime(2026, 1, 1, tzinfo=timezone.utc)
FECHA_FIN    = datetime(2026, 4, 2, 23, 59, 59, tzinfo=timezone.utc)

API_BASE  = "https://arctic-shift.photon-reddit.com/api"
ENDPOINT  = f"{API_BASE}/posts/search"
LIMIT_POR_PAGINA = 100
PAUSA_SEGUNDOS   = 0.5

DIR_REDDIT = os.path.join("data", "raw", "reddit")

COLUMNAS = [
    "id", "created_utc", "fecha", "titulo", "texto",
    "autor", "score", "num_comentarios", "upvote_ratio",
    "subreddit", "url", "ticker",
]


# ── Funciones — reutilizamos la misma lógica que en download_reddit.py ───────

def ts_a_fecha(ts) -> str:
    """Convierte un timestamp Unix a una fecha legible en formato YYYY-MM-DD."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def parsear_post(post: dict, ticker: str) -> dict:
    """Toma el JSON de un post de Arctic Shift y extrae solo los campos que
    nos interesan, limpiando saltos de línea y espacios de más."""
    return {
        "id":               post.get("id", ""),
        "created_utc":      post.get("created_utc", ""),
        "fecha":            ts_a_fecha(post.get("created_utc")),
        "titulo":           post.get("title", "").replace("\n", " ").strip(),
        "texto":            post.get("selftext", "").replace("\n", " ").strip(),
        "autor":            post.get("author", ""),
        "score":            post.get("score", 0),
        "num_comentarios":  post.get("num_comments", 0),
        "upvote_ratio":     post.get("upvote_ratio", None),
        "subreddit":        post.get("subreddit", ""),
        "url":              post.get("url", ""),
        "ticker":           ticker,
    }


def llamar_api(cashtag: str, subreddit: str, after_ts: int, before_ts: int) -> list:
    """Hace una llamada al endpoint de Arctic Shift buscando posts con un cashtag
    en un subreddit concreto. Si falla, reintenta hasta 3 veces con pausas."""
    params = {
        "query": cashtag, "subreddit": subreddit,
        "after": after_ts, "before": before_ts,
        "limit": LIMIT_POR_PAGINA, "sort": "asc",
    }
    for intento in range(3):
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                print(f"    [429] Rate limit. Esperando 60s...")
                time.sleep(60)
            else:
                print(f"    [HTTP {resp.status_code}] Reintentando en 10s...")
                time.sleep(10)
        except requests.exceptions.RequestException as e:
            print(f"    [ERROR RED] {e}. Reintentando en 10s...")
            time.sleep(10)
    return []


def descargar_combinacion(ticker: str, cashtag: str, subreddit: str) -> list:
    """Descarga todos los posts de un ticker+subreddit para el periodo ene-mar 2026.
    Usa paginación temporal igual que el script original: avanza el cursor al último
    timestamp recibido hasta agotar los resultados."""
    after_ts  = int(FECHA_INICIO.timestamp())
    before_ts = int(FECHA_FIN.timestamp())
    todos = []
    pagina = 0

    while after_ts < before_ts:
        pagina += 1
        posts = llamar_api(cashtag, subreddit, after_ts, before_ts)
        if not posts:
            break
        nuevos = [parsear_post(p, ticker) for p in posts]
        todos.extend(nuevos)
        ultimo_ts = int(posts[-1].get("created_utc", after_ts))
        if ultimo_ts <= after_ts:
            break
        after_ts = ultimo_ts + 1
        print(f"    Pág {pagina:3d} | +{len(nuevos):3d} | Total: {len(todos):4d} | Hasta: {ts_a_fecha(ultimo_ts)}")
        time.sleep(PAUSA_SEGUNDOS)
        if len(posts) < LIMIT_POR_PAGINA:
            break
    return todos


# ── Main — descarga incremental y append a los CSVs existentes ───────────────

def main():
    os.makedirs(DIR_REDDIT, exist_ok=True)

    print("=" * 60)
    print("  Reddit Update — ene-mar 2026 (Arctic Shift)")
    print(f"  Periodo : {FECHA_INICIO.date()} → {FECHA_FIN.date()}")
    print(f"  Tickers : {', '.join(TICKERS.keys())}")
    print("=" * 60)

    resumen = {}

    for ticker, cashtag in TICKERS.items():
        print(f"\n{'─'*60}")
        print(f"  {ticker} ({cashtag})")
        print(f"{'─'*60}")

        # Cargamos el CSV que ya teníamos para poder deduplicar contra los posts nuevos
        ruta_csv = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2025.csv")
        if os.path.exists(ruta_csv):
            df_existente = pd.read_csv(ruta_csv, engine="python", on_bad_lines="skip")
            ids_existentes = set(df_existente["id"].astype(str))
            n_antes = len(df_existente)
        else:
            df_existente = pd.DataFrame(columns=COLUMNAS)
            ids_existentes = set()
            n_antes = 0

        # Descargamos de todos los subreddits los posts del periodo 2026
        posts_nuevos = []
        for subreddit in SUBREDDITS:
            print(f"\n  r/{subreddit}")
            posts = descargar_combinacion(ticker, cashtag, subreddit)
            unicos = [p for p in posts if str(p["id"]) not in ids_existentes]
            ids_existentes.update(str(p["id"]) for p in unicos)
            posts_nuevos.extend(unicos)
            print(f"  → {len(posts)} posts, {len(unicos)} nuevos únicos")

        if posts_nuevos:
            df_nuevos = pd.DataFrame(posts_nuevos, columns=COLUMNAS)
            df_nuevos["created_utc"] = pd.to_numeric(df_nuevos["created_utc"], errors="coerce")

            # Juntamos los posts nuevos con los existentes y guardamos
            df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
            df_final = df_final.sort_values("created_utc").reset_index(drop=True)

            # Guardamos con el nombre actualizado (2019_2026)
            ruta_nueva = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2026.csv")
            df_final.to_csv(ruta_nueva, index=False, encoding="utf-8")
            # También guardamos con el nombre viejo para que scripts que lo referencien sigan funcionando
            df_final.to_csv(ruta_csv, index=False, encoding="utf-8")

            n_despues = len(df_final)
            print(f"\n  {ticker}: {n_antes:,} → {n_despues:,} posts (+{n_despues-n_antes:,} nuevos)")
            print(f"  Guardado: {ruta_nueva}")
            resumen[ticker] = (n_antes, n_despues)
        else:
            print(f"\n  {ticker}: sin posts nuevos en ene-mar 2026")
            resumen[ticker] = (n_antes, n_antes)

    # Generamos el CSV combinado juntando todos los tickers
    print(f"\n{'='*60}")
    print("  Generando CSV combinado...")
    frames = []
    for ticker in TICKERS:
        ruta = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2026.csv")
        if not os.path.exists(ruta):
            ruta = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2025.csv")
        if os.path.exists(ruta):
            frames.append(pd.read_csv(ruta, engine="python", on_bad_lines="skip"))
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.sort_values(["ticker", "created_utc"]).reset_index(drop=True)
        ruta_all = os.path.join(DIR_REDDIT, "ALL_reddit_2019_2026.csv")
        df_all.to_csv(ruta_all, index=False, encoding="utf-8")
        print(f"  {ruta_all}: {len(df_all):,} posts totales")

    # Resumen final de cuántos posts se añadieron por cada empresa
    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    for ticker, (antes, despues) in resumen.items():
        print(f"  {ticker:5s}: {antes:>6,} → {despues:>6,}  (+{despues-antes:,})")
    print(f"\n  Actualización completada.")


if __name__ == "__main__":
    main()
