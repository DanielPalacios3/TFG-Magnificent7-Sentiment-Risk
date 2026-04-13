"""
Script: download_reddit.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Descarga todos los posts históricos de Reddit que mencionan a las 7 empresas del
grupo Magnificent 7. Para ello usa la API pública de Arctic Shift
(https://arctic-shift.photon-reddit.com), que es un archivo masivo de Reddit.
La búsqueda se hace por cashtag ($AAPL, $TSLA, etc.) en los subreddits de finanzas
más activos.

¿Por qué usamos Arctic Shift en vez de PRAW (la librería oficial de Reddit)?
-----------------------------------------------------------------------------
Lo primero que probamos fue PRAW, pero resulta que solo te deja acceder a los
~1.000 posts más recientes de cada subreddit. Para el periodo 2024-2025 que
necesitamos eso se queda muy corto. Arctic Shift en cambio es un proyecto que
archiva todo Reddit desde 2005, y expone una API REST completamente gratuita,
sin credenciales ni registro. Perfecto para nuestro caso.

¿Y por qué buscamos por cashtag ($AAPL) en vez de por nombre de empresa?
-------------------------------------------------------------------------
En las comunidades financieras de Reddit es estándar usar el cashtag ($AAPL, $NVDA...)
cuando se habla de una acción concreta. Si buscásemos "Apple" a secas nos saldrían
un montón de resultados sobre Apple Music, la fruta, etc. El cashtag es mucho más
preciso y evita ese ruido.

¿Qué archivos genera?
---------------------
- data/raw/reddit/{TICKER}_reddit_2019_2025.csv  → un CSV por cada empresa
- data/raw/reddit/ALL_reddit_2019_2025.csv       → un CSV combinado con todas juntas

Columnas que tiene cada CSV:
- id              : identificador único del post (lo usamos para deduplicar)
- created_utc     : timestamp Unix del momento de creación
- fecha           : la fecha en formato legible (YYYY-MM-DD)
- titulo          : el título del post tal cual lo escribió el usuario
- texto           : el cuerpo del post (selftext); está vacío en posts que son solo links
- autor           : nombre de usuario del autor
- score           : puntuación neta del post (upvotes menos downvotes)
- num_comentarios : cuántos comentarios tiene el post
- upvote_ratio    : proporción de votos positivos (va de 0 a 1)
- subreddit       : de qué subreddit viene el post
- url             : la URL directa al post
- ticker          : el ticker de la empresa (esto lo añade el script, no viene de Reddit)

Uso
---
    source venv/bin/activate
    python scripts/download_reddit.py
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ── Configuración — parámetros principales de la descarga ────────────────────

# Cada ticker con su cashtag correspondiente, que es lo que buscamos en Reddit
TICKERS = {
    "AAPL":  "$AAPL",
    "MSFT":  "$MSFT",
    "GOOGL": "$GOOGL",
    "AMZN":  "$AMZN",
    "NVDA":  "$NVDA",
    "META":  "$META",
    "TSLA":  "$TSLA",
}

# Los subreddits que consultamos, elegidos por ser los más activos en finanzas minoristas.
# Añadimos StockMarket y options después de comprobar que tenían buen volumen en Arctic Shift.
# r/SecurityAnalysis lo descartamos porque apenas tenía menciones con formato $TICKER.
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
]

# El rango de fechas que queremos cubrir
FECHA_INICIO = datetime(2019, 1, 1, tzinfo=timezone.utc)
FECHA_FIN    = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Configuración de la API de Arctic Shift
API_BASE  = "https://arctic-shift.photon-reddit.com/api"
ENDPOINT  = f"{API_BASE}/posts/search"
LIMIT_POR_PAGINA = 100   # es el máximo que permite la API por llamada

# Pausa entre llamadas para no saturar Arctic Shift. No publican un rate limit
# estricto, pero 0.5 segundos va bien y es respetuoso con su servidor.
PAUSA_SEGUNDOS = 0.5

# Donde guardamos los CSVs resultantes
DIR_REDDIT = os.path.join("data", "raw", "reddit")

# Las columnas que tendrá el CSV final, en este orden
COLUMNAS = [
    "id", "created_utc", "fecha", "titulo", "texto",
    "autor", "score", "num_comentarios", "upvote_ratio",
    "subreddit", "url", "ticker",
]

# ── Funciones auxiliares — las piezas que hacen el trabajo pesado ─────────────


def ts_a_fecha(ts) -> str:
    """Convierte un timestamp Unix (esos números enormes tipo 1704067200) a una
    fecha legible en formato YYYY-MM-DD. Si el timestamp viene mal, devuelve vacío."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def parsear_post(post: dict, ticker: str) -> dict:
    """Toma el JSON crudo de un post que devuelve Arctic Shift y extrae solo los
    campos que nos interesan, limpiando saltos de línea y espacios sobrantes."""
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


def llamar_api(cashtag: str, subreddit: str, after_ts: int, before_ts: int) -> list[dict]:
    """Hace una llamada al endpoint de búsqueda de Arctic Shift pidiendo posts que
    contengan el cashtag en un subreddit concreto, dentro de un rango temporal dado.
    Los resultados vienen ordenados del más antiguo al más reciente (ascendente).

    Si la API falla (error de red, rate limit, etc.), reintenta hasta 3 veces con
    pausas progresivas. Si después de 3 intentos no hay respuesta, devuelve lista vacía.
    """
    params = {
        "query":     cashtag,
        "subreddit": subreddit,
        "after":     after_ts,
        "before":    before_ts,
        "limit":     LIMIT_POR_PAGINA,
        "sort":      "asc",           # ascendente: del más antiguo al más reciente
    }

    for intento in range(3):
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                # Arctic Shift mete los posts dentro de la clave "data" del JSON
                return data.get("data", [])

            elif resp.status_code == 429:
                print(f"    [429] Rate limit. Esperando 60s...")
                time.sleep(60)
                continue

            else:
                print(f"    [HTTP {resp.status_code}] Reintentando en 10s...")
                time.sleep(10)
                continue

        except requests.exceptions.RequestException as e:
            print(f"    [ERROR RED] {e}. Reintentando en 10s...")
            time.sleep(10)

    print(f"    [FALLO] No se pudo obtener respuesta tras 3 intentos.")
    return []


def descargar_combinacion(ticker: str, cashtag: str, subreddit: str) -> list[dict]:
    """Descarga todos los posts de una combinación ticker+subreddit para todo el
    periodo configurado. Como Arctic Shift no tiene paginación por cursor, lo que
    hacemos es una paginación temporal: después de cada página, movemos el after_ts
    al timestamp del último post que recibimos + 1 segundo, y seguimos pidiendo
    hasta que no queden más resultados o lleguemos al final del periodo.
    """
    after_ts  = int(FECHA_INICIO.timestamp())
    before_ts = int(FECHA_FIN.timestamp())

    todos = []
    pagina = 0

    while after_ts < before_ts:
        pagina += 1
        posts = llamar_api(cashtag, subreddit, after_ts, before_ts)

        if not posts:
            break  # ya no quedan más resultados en este rango temporal

        nuevos = [parsear_post(p, ticker) for p in posts]
        todos.extend(nuevos)

        # Movemos el cursor temporal al timestamp del último post que nos llegó
        ultimo_ts = int(posts[-1].get("created_utc", after_ts))
        if ultimo_ts <= after_ts:
            break  # si no avanzamos, cortamos para no quedarnos en bucle infinito
        after_ts = ultimo_ts + 1

        fecha_ultimo = ts_a_fecha(ultimo_ts)
        print(
            f"    Página {pagina:3d} | "
            f"+{len(nuevos):3d} posts | "
            f"Total: {len(todos):5,d} | "
            f"Hasta: {fecha_ultimo}"
        )

        time.sleep(PAUSA_SEGUNDOS)

        # Si la página vino con menos posts del máximo, significa que ya no quedan más
        if len(posts) < LIMIT_POR_PAGINA:
            break

    return todos


def descargar_ticker(ticker: str, cashtag: str) -> list[dict]:
    """Recorre todos los subreddits configurados descargando posts para un ticker
    concreto. Deduplica por ID de post para que no haya repetidos (aunque en
    principio cada subreddit es independiente, por si acaso)."""
    print(f"\n{'='*60}")
    print(f"  Ticker: {ticker}  |  Cashtag: {cashtag}")
    print(f"  Periodo: {FECHA_INICIO.date()} → {FECHA_FIN.date()}")
    print(f"{'='*60}")

    todos = []
    ids_vistos = set()

    for subreddit in SUBREDDITS:
        print(f"\n  Subreddit: r/{subreddit}")
        posts = descargar_combinacion(ticker, cashtag, subreddit)

        # Deduplicamos por ID del post por si aparece en múltiples búsquedas
        nuevos_unicos = [p for p in posts if p["id"] not in ids_vistos]
        ids_vistos.update(p["id"] for p in nuevos_unicos)
        todos.extend(nuevos_unicos)

        print(f"  r/{subreddit}: {len(posts):,d} posts ({len(nuevos_unicos):,d} únicos)")

    print(f"\n  Total único para {ticker}: {len(todos):,d} posts")
    return todos


def guardar_csv(posts: list[dict], ticker: str) -> str:
    """Toma la lista de posts de un ticker, los mete en un DataFrame, los ordena
    cronológicamente y los guarda como CSV. Devuelve la ruta del archivo creado."""
    if not posts:
        print(f"  Sin posts para {ticker}.")
        return ""

    ruta = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2025.csv")
    df = pd.DataFrame(posts, columns=COLUMNAS)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.sort_values("created_utc").reset_index(drop=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"  Guardado: {ruta}  ({len(df):,d} posts)")
    return ruta


# ── Ejecución principal — orquesta la descarga de todos los tickers ───────────

def main():
    os.makedirs(DIR_REDDIT, exist_ok=True)

    print("=" * 60)
    print("  Reddit Downloader (Arctic Shift) — TFG Magnificent 7")
    print(f"  Periodo : {FECHA_INICIO.date()} → {FECHA_FIN.date()}")
    print(f"  Tickers : {', '.join(TICKERS.keys())}")
    print(f"  Subreddits: {', '.join('r/' + s for s in SUBREDDITS)}")
    print("=" * 60)

    todos_los_posts = []

    for ticker, cashtag in TICKERS.items():
        ruta_csv = os.path.join(DIR_REDDIT, f"{ticker}_reddit_2019_2025.csv")

        # Lógica de reanudación: si ya tenemos un CSV para este ticker, miramos qué
        # subreddits ya están descargados y solo bajamos los que faltan
        posts_existentes = []
        subreddits_pendientes = list(SUBREDDITS)

        if os.path.exists(ruta_csv):
            df_existente = pd.read_csv(ruta_csv, engine="python", on_bad_lines="skip")
            subs_descargados = set(df_existente["subreddit"].unique())
            subreddits_pendientes = [s for s in SUBREDDITS if s not in subs_descargados]
            posts_existentes = df_existente.to_dict("records")

            if not subreddits_pendientes:
                print(f"\n[{ticker}] Completo ({len(posts_existentes):,} posts, "
                      f"subreddits: {sorted(subs_descargados)}). Saltando.")
                todos_los_posts.extend(posts_existentes)
                continue

            print(f"\n[{ticker}] CSV existente ({len(posts_existentes):,} posts). "
                  f"Subreddits pendientes: {subreddits_pendientes}")

        # Solo descargamos los subreddits que todavía no teníamos
        posts_nuevos = []
        ids_vistos = {p["id"] for p in posts_existentes}

        for subreddit in subreddits_pendientes:
            print(f"\n  Subreddit: r/{subreddit}")
            posts = descargar_combinacion(ticker, cashtag, subreddit)
            nuevos_unicos = [p for p in posts if p["id"] not in ids_vistos]
            ids_vistos.update(p["id"] for p in nuevos_unicos)
            posts_nuevos.extend(nuevos_unicos)
            print(f"  r/{subreddit}: {len(posts):,} posts ({len(nuevos_unicos):,} únicos nuevos)")

        todos_posts_ticker = posts_existentes + posts_nuevos
        print(f"\n  Total para {ticker}: {len(todos_posts_ticker):,} posts")

        if todos_posts_ticker:
            guardar_csv(todos_posts_ticker, ticker)
            todos_los_posts.extend(todos_posts_ticker)

    # Juntamos todos los posts de todas las empresas en un solo CSV combinado
    if todos_los_posts:
        ruta_all = os.path.join(DIR_REDDIT, "ALL_reddit_2019_2025.csv")
        df_all = pd.DataFrame(todos_los_posts)
        # Nos aseguramos de que todas las columnas existan aunque haya mezcla de fuentes
        for col in COLUMNAS:
            if col not in df_all.columns:
                df_all[col] = None
        df_all = df_all[COLUMNAS].sort_values(["ticker", "created_utc"]).reset_index(drop=True)
        df_all.to_csv(ruta_all, index=False, encoding="utf-8")
        print(f"\n  CSV combinado: {ruta_all}  ({len(df_all):,d} posts totales)")

    print("\n  Descarga completada.")
    print(f"  Archivos en: {DIR_REDDIT}/")


if __name__ == "__main__":
    main()
