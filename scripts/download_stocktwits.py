"""
Script: download_stocktwits.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Este script descarga mensajes históricos de StockTwits para las 7 empresas del
grupo Magnificent 7. Usa la API pública gratuita de StockTwits, que no necesita
API key. La forma de recorrer el historial es paginar hacia atrás en el tiempo
usando el parámetro `max` (que funciona por ID numérico de mensaje), y se va
parando cuando llega a la fecha límite que le configuremos (en nuestro caso,
1 de enero de 2024).

¿Por qué StockTwits y no Twitter?
----------------------------------
StockTwits es una red social 100% financiera, como un Twitter pero solo para
inversores. La usamos como sustituto de Twitter/X porque desde 2023 Twitter
cerró el grifo del acceso gratuito a datos históricos. StockTwits en cambio
tiene una API pública de solo lectura que no pide ni registro ni API key, así
que es nuestra mejor alternativa para capturar el sentimiento de inversores
minoristas.

¿Qué archivos genera?
---------------------
- data/raw/social/{TICKER}_stocktwits_2024_2026.csv  → un CSV por empresa
- data/raw/social/ALL_stocktwits_2024_2026.csv       → CSV combinado con todas
- data/raw/social/checkpoints.json                   → guarda el último ID
                                                        procesado por ticker,
                                                        para poder reanudar si
                                                        se interrumpe la descarga

¿Para qué sirve cada campo que guardamos?
-----------------------------------------
- id            : el identificador único del mensaje, que además es la clave para
                  paginar hacia atrás (la API usa IDs, no fechas)
- created_at    : fecha y hora UTC del mensaje, para poder alinearlo con el mercado
- body          : el texto del mensaje en sí, que luego pasaremos por FinBERT
- sentiment     : la etiqueta bullish/bearish que pone el propio usuario al publicar
                  (solo ~40-50% de los mensajes la incluyen, pero es oro puro como señal)
- username      : quién escribió el mensaje
- followers     : cuántos seguidores tiene el usuario, por si queremos ponderar
- likes_count   : cuántos likes tiene el mensaje, como proxy de su relevancia
- ticker        : la empresa a la que corresponde el mensaje

Ojo con las limitaciones de la API gratuita
-------------------------------------------
- Sin autenticación solo se pueden hacer 200 llamadas por hora, y cada una trae
  como máximo 30 mensajes. Eso da unas ~138 horas de ejecución para 7 tickers.
- No hay forma de filtrar por fecha directamente — la paginación va por ID.
- La buena noticia es que el script es REANUDABLE: va guardando checkpoints con
  el último ID procesado por ticker. Si se corta, la próxima vez retoma donde
  se quedó automáticamente.
- El periodo 2024-2026 no es casualidad: cubre eventos clave como el boom de la
  IA en 2024, el ciclo electoral de EEUU y las fuertes fluctuaciones de NVDA/META/TSLA.

Uso
---
    python scripts/download_stocktwits.py

    Si quieres forzar que re-descargue un ticker concreto, borra su entrada
    del archivo data/raw/social/checkpoints.json y vuelve a ejecutar.
"""

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ── Configuración — parámetros de la descarga ────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Hasta dónde paginamos hacia atrás: cuando los mensajes sean anteriores a esta
# fecha, paramos. No necesitamos ir más atrás de enero 2024 para este TFG.
FECHA_INICIO = datetime(2024, 1, 1, tzinfo=timezone.utc)

# La URL base de la API pública de StockTwits (no hace falta API key para leer)
API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

# Cuánto esperamos entre cada llamada a la API. El rate limit es 200 req/hora,
# que sale a 1 petición cada 18 segundos. Ponemos 20 para tener margen y no
# arriesgarnos a un HTTP 429 (Too Many Requests).
PAUSA_SEGUNDOS = 20

# Donde guardamos todo: CSVs y el archivo de checkpoints
DIR_SOCIAL = os.path.join("data", "raw", "social")
CHECKPOINT_FILE = os.path.join(DIR_SOCIAL, "checkpoints.json")

# Las columnas que tendrá cada CSV
COLUMNAS = [
    "id",
    "created_at",
    "body",
    "sentiment",
    "username",
    "followers",
    "likes_count",
    "ticker",
]

# ── Funciones auxiliares — cada una hace una tarea concreta ──────────────────


def cargar_checkpoints() -> dict:
    """Lee el fichero de checkpoints, que guarda el último ID procesado para
    cada ticker. Si es la primera ejecución y el archivo no existe todavía,
    simplemente devuelve un diccionario vacío."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_checkpoint(checkpoints: dict, ticker: str, max_id: int) -> None:
    """Actualiza el checkpoint de un ticker y lo escribe a disco inmediatamente.
    Así, si el script se interrumpe, la próxima vez retoma desde aquí."""
    checkpoints[ticker] = max_id
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)


def parsear_mensaje(msg: dict, ticker: str) -> dict:
    """Toma el JSON crudo de un mensaje de StockTwits y extrae solo los campos
    que nos interesan para el análisis de sentimiento."""
    # El sentimiento solo está presente si el usuario lo eligió al publicar
    # (Bullish o Bearish). Muchos mensajes no lo traen, y eso está bien.
    entities = msg.get("entities") or {}
    sentiment_obj = entities.get("sentiment") or {}
    sentiment = sentiment_obj.get("basic")  # "Bullish", "Bearish" o None

    usuario = msg.get("user") or {}
    likes_obj = msg.get("likes") or {}

    return {
        "id": msg.get("id"),
        "created_at": msg.get("created_at"),
        "body": msg.get("body", "").replace("\n", " ").strip(),
        "sentiment": sentiment,
        "username": usuario.get("username"),
        "followers": usuario.get("followers"),
        "likes_count": likes_obj.get("total", 0),
        "ticker": ticker,
    }


def llamar_api(ticker: str, max_id: int | None = None) -> dict | None:
    """Hace una llamada a la API de StockTwits y devuelve el JSON de respuesta.

    Parámetros
    ----------
    ticker : el símbolo bursátil de la empresa, por ejemplo 'AAPL'.
    max_id : si lo pasamos, la API nos devuelve solo mensajes con ID menor que
             este valor, que es la forma de paginar hacia atrás en el tiempo.

    Si recibimos un HTTP 429 (rate limit), esperamos el tiempo que nos dice el
    header de la respuesta y reintentamos. Si el ticker no existe (404), devuelve
    None para que el script pase al siguiente.
    """
    url = API_URL.format(ticker=ticker)
    params = {"limit": 30}
    if max_id is not None:
        params["max"] = max_id

    for intento in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 200:
                return resp.json()

            elif resp.status_code == 429:
                # Nos han cortado por exceder el rate limit, toca esperar
                reset_ts = resp.headers.get("X-RateLimit-Reset")
                if reset_ts:
                    espera = max(int(reset_ts) - int(time.time()) + 5, 60)
                else:
                    espera = 3600  # si no viene el header, esperamos 1 hora por precaución
                print(f"  [429] Rate limit alcanzado. Esperando {espera:.0f}s...")
                time.sleep(espera)
                continue

            elif resp.status_code == 404:
                print(f"  [404] Ticker {ticker} no encontrado en StockTwits.")
                return None

            else:
                print(f"  [HTTP {resp.status_code}] Error inesperado. Reintentando...")
                time.sleep(30)
                continue

        except requests.exceptions.RequestException as e:
            print(f"  [ERROR RED] {e}. Reintentando en 30s...")
            time.sleep(30)

    print(f"  [FALLO] No se pudo obtener respuesta para {ticker} tras 3 intentos.")
    return None


def descargar_ticker(ticker: str, checkpoints: dict) -> list[dict]:
    """Descarga todos los mensajes de StockTwits para un ticker desde FECHA_INICIO
    hasta hoy. Va paginando hacia atrás en el tiempo usando el parámetro `max` de
    la API (que funciona por ID de mensaje, no por fecha).

    El script se detiene cuando pasa cualquiera de estas tres cosas:
    - El mensaje más antiguo de la página es anterior a FECHA_INICIO
    - La API dice que ya no quedan más mensajes (cursor.more = False)
    - Hay un error que no se puede recuperar

    Los checkpoints se guardan constantemente, así que si se corta la descarga
    (que con ~138 horas de ejecución total es muy probable), la próxima vez
    retoma justo donde se quedó.
    """
    print(f"\n{'='*60}")
    print(f"  Ticker: {ticker}")
    print(f"  Descargando mensajes desde {FECHA_INICIO.date()} hasta hoy")
    print(f"{'='*60}")

    mensajes = []
    max_id = checkpoints.get(ticker)  # None en primera ejecución

    if max_id:
        print(f"  Reanudando desde checkpoint: ID < {max_id}")
    else:
        print(f"  Primera descarga — empezando desde el mensaje más reciente")

    pagina = 0
    total_descargados = 0

    while True:
        pagina += 1
        datos = llamar_api(ticker, max_id=max_id)

        if datos is None:
            print(f"  Descarga interrumpida en página {pagina}.")
            break

        # Comprobamos que la API nos ha respondido correctamente
        estado = (datos.get("response") or {}).get("status", 0)
        if estado != 200:
            print(f"  API devolvió estado {estado}. Parando.")
            break

        mensajes_pagina = datos.get("messages", [])
        if not mensajes_pagina:
            print(f"  Sin mensajes en página {pagina}. Fin de la paginación.")
            break

        # Procesamos cada mensaje de la página y comprobamos si sigue dentro de nuestro rango
        mensajes_validos = []
        fecha_mas_antigua = None

        for msg in mensajes_pagina:
            created_raw = msg.get("created_at", "")
            try:
                # El formato habitual de la API es "2024-03-15T14:30:00Z"
                fecha_msg = datetime.strptime(created_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                # Probamos con formato ISO por si viene con offset horario distinto
                try:
                    fecha_msg = datetime.fromisoformat(created_raw)
                    if fecha_msg.tzinfo is None:
                        fecha_msg = fecha_msg.replace(tzinfo=timezone.utc)
                except ValueError:
                    fecha_msg = None

            if fecha_msg is None:
                continue

            if fecha_mas_antigua is None or fecha_msg < fecha_mas_antigua:
                fecha_mas_antigua = fecha_msg

            # Solo nos quedamos con mensajes que caigan dentro de nuestro periodo
            if fecha_msg >= FECHA_INICIO:
                mensajes_validos.append(parsear_mensaje(msg, ticker))

        total_descargados += len(mensajes_validos)
        mensajes.extend(mensajes_validos)

        # Sacamos el cursor que nos dice cómo pedir la siguiente página
        cursor = datos.get("cursor") or {}
        nuevo_max_id = cursor.get("max")
        hay_mas = cursor.get("more", False)

        # Mostramos el progreso para saber por dónde va
        fecha_str = fecha_mas_antigua.strftime("%Y-%m-%d") if fecha_mas_antigua else "?"
        print(
            f"  Página {pagina:4d} | "
            f"Válidos: {len(mensajes_validos):3d} | "
            f"Total acumulado: {total_descargados:7,d} | "
            f"Fecha más antigua: {fecha_str}"
        )

        # Si ya hemos llegado a antes de la fecha límite, paramos
        if fecha_mas_antigua and fecha_mas_antigua < FECHA_INICIO:
            print(f"\n  Alcanzada fecha límite ({FECHA_INICIO.date()}). Parando.")
            break

        # Si la API dice que no quedan más mensajes, hemos llegado al final
        if not hay_mas or nuevo_max_id is None:
            print(f"\n  API indica que no hay más mensajes. Fin del histórico.")
            break

        # Preparamos la siguiente iteración y guardamos checkpoint por si se corta
        max_id = nuevo_max_id
        guardar_checkpoint(checkpoints, ticker, max_id)

        # Esperamos para respetar el rate limit de 200 peticiones por hora
        time.sleep(PAUSA_SEGUNDOS)

    print(f"\n  Total mensajes descargados para {ticker}: {total_descargados:,d}")
    return mensajes


def guardar_csv(mensajes: list[dict], ticker: str) -> str:
    """Mete los mensajes de un ticker en un DataFrame, los ordena de más antiguo
    a más reciente y los guarda como CSV. Devuelve la ruta del archivo creado."""
    if not mensajes:
        print(f"  Sin mensajes para guardar en {ticker}.")
        return ""

    ruta = os.path.join(DIR_SOCIAL, f"{ticker}_stocktwits_2024_2026.csv")
    df = pd.DataFrame(mensajes, columns=COLUMNAS)

    # Ordenamos cronológicamente: el más antiguo primero
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.sort_values("created_at").reset_index(drop=True)

    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"  Guardado: {ruta} ({len(df):,d} mensajes)")
    return ruta


# ── Ejecución principal — desde aquí se lanza todo ──────────────────────────

def main():
    """Función principal que recorre los 7 tickers, descarga los mensajes de
    StockTwits para cada uno y al final genera un CSV combinado con todo."""
    os.makedirs(DIR_SOCIAL, exist_ok=True)

    print("=" * 60)
    print("  StockTwits Downloader — TFG Magnificent 7")
    print(f"  Periodo: {FECHA_INICIO.date()} → hoy")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Pausa entre llamadas: {PAUSA_SEGUNDOS}s")
    print(f"  Throughput: ~{3600 // PAUSA_SEGUNDOS * 30:,d} mensajes/hora")
    print("=" * 60)

    checkpoints = cargar_checkpoints()
    todos_los_mensajes = []

    for ticker in TICKERS:
        mensajes_ticker = descargar_ticker(ticker, checkpoints)

        if mensajes_ticker:
            guardar_csv(mensajes_ticker, ticker)
            todos_los_mensajes.extend(mensajes_ticker)

        # El checkpoint final del ticker ya se fue guardando incrementalmente
        # durante la descarga, así que no hace falta hacer nada más aquí

    # Juntamos todos los mensajes de todas las empresas en un solo CSV
    if todos_los_mensajes:
        ruta_all = os.path.join(DIR_SOCIAL, "ALL_stocktwits_2024_2026.csv")
        df_all = pd.DataFrame(todos_los_mensajes, columns=COLUMNAS)
        df_all["created_at"] = pd.to_datetime(df_all["created_at"], utc=True, errors="coerce")
        df_all = df_all.sort_values(["ticker", "created_at"]).reset_index(drop=True)
        df_all.to_csv(ruta_all, index=False, encoding="utf-8")
        print(f"\n  CSV combinado: {ruta_all} ({len(df_all):,d} mensajes totales)")

    print("\n  Descarga finalizada.")
    print(f"  Archivos en: {DIR_SOCIAL}/")


if __name__ == "__main__":
    main()
