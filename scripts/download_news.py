"""
Script: download_news.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace?
----------
Extrae artículos de noticias financieras de GDELT Project (API v2 Doc)
para las 7 empresas Magnificent 7 durante el periodo 2019-01-01 a 2024-12-31.
Los datos se guardan en CRUDO, sin ningún tipo de limpieza ni filtrado,
para ser procesados posteriormente en la Fase 3 (NLP + FinBERT).

La API de GDELT devuelve hasta 250 artículos por consulta, por lo que
se hacen consultas mensuales por empresa para maximizar la cobertura.

¿Qué genera?
------------
- data/raw/news/{TICKER}_gdelt_2019_2024.csv  → un CSV por empresa
- data/raw/news/ALL_gdelt_2019_2024.csv       → CSV combinado

¿Por qué GDELT?
---------------
GDELT (Global Database of Events, Language, and Tone) es una base de datos
global de noticias completamente gratuita y sin necesidad de API key.
Calcula automáticamente el tono (sentimiento) de cada artículo mediante
su sistema de análisis de texto, lo que nos proporciona una señal de
sentimiento pre-calculada que complementará el análisis con FinBERT.

Campos extraídos (en crudo):
- seendate         : fecha y hora de publicación del artículo
- title            : titular del artículo
- url              : URL original del artículo
- domain           : dominio del medio de comunicación
- language         : idioma detectado por GDELT
- sourcecountry    : país de origen del medio
- tone             : tono global del artículo [-100, +100] (GDELT V2Tone[0])
- score_positivo   : puntuación de densidad de palabras positivas (V2Tone[1])
- score_negativo   : puntuación de densidad de palabras negativas (V2Tone[2])
- polaridad        : diferencia entre densidad positiva y negativa (V2Tone[3])
- num_palabras     : longitud del artículo en palabras (V2Tone[6])
- query_ticker     : ticker por el que se encontró el artículo
- query_empresa    : nombre de empresa buscado en la consulta
- query_mes        : mes de la consulta (YYYY-MM), para trazabilidad
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Términos de búsqueda por ticker: se usan los nombres conocidos de cada empresa
# para maximizar la cobertura de artículos relevantes
EMPRESAS = {
    "AAPL":  '"Apple" sourcelang:english',
    "MSFT":  '"Microsoft" sourcelang:english',
    "GOOGL": '"Alphabet" OR "Google" sourcelang:english',
    "AMZN":  '"Amazon" sourcelang:english',
    "NVDA":  '"NVIDIA" sourcelang:english',
    "META":  '"Meta" OR "Facebook" sourcelang:english',
    "TSLA":  '"Tesla" sourcelang:english',
}

FECHA_INICIO  = datetime(2019, 1, 1)
FECHA_FIN     = datetime(2024, 12, 31)
PAUSA_SEGUNDOS = 5.5   # pausa entre llamadas para respetar la API pública (evitar 429)
MAX_REGISTROS  = 250   # límite de la API GDELT por consulta

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_SALIDA    = RAIZ_PROYECTO / "data" / "raw" / "news"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def generar_meses(inicio: datetime, fin: datetime) -> list[tuple[str, str]]:
    """
    Genera lista de (inicio_mes, fin_mes) en formato YYYYMMDDHHMMSS
    para iterar mes a mes entre las fechas dadas.
    """
    meses = []
    actual = inicio.replace(day=1)
    while actual <= fin:
        fin_mes = actual + relativedelta(months=1) - relativedelta(days=1)
        fin_mes = min(fin_mes, fin)
        meses.append((
            actual.strftime("%Y%m%d000000"),
            fin_mes.strftime("%Y%m%d235959"),
        ))
        actual += relativedelta(months=1)
    return meses


def parsear_v2tone(v2tone_str: str) -> dict:
    """
    Parsea el campo V2Tone de GDELT, que es una cadena CSV con 7 valores:
    [0] tono global, [1] densidad positiva, [2] densidad negativa,
    [3] polaridad, [4] densidad ref. actividad, [5] densidad auto-referencia,
    [6] número de palabras.

    Devuelve un dict con los campos relevantes para el TFG.
    """
    try:
        partes = [float(x) for x in v2tone_str.split(",")]
        return {
            "tone":          partes[0] if len(partes) > 0 else None,
            "score_positivo": partes[1] if len(partes) > 1 else None,
            "score_negativo": partes[2] if len(partes) > 2 else None,
            "polaridad":     partes[3] if len(partes) > 3 else None,
            "num_palabras":  int(partes[6]) if len(partes) > 6 else None,
        }
    except Exception:
        return {"tone": None, "score_positivo": None, "score_negativo": None,
                "polaridad": None, "num_palabras": None}


def consultar_gdelt(query: str, inicio: str, fin: str) -> list[dict]:
    """
    Realiza una consulta a la API GDELT v2 Doc y devuelve lista de artículos.

    Parámetros
    ----------
    query  : término de búsqueda en formato GDELT
    inicio : fecha inicio en formato YYYYMMDDHHMMSS
    fin    : fecha fin en formato YYYYMMDDHHMMSS

    Devuelve
    --------
    Lista de dicts con los campos de cada artículo, o lista vacía si falla.
    """
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    MAX_REGISTROS,
        "format":        "json",
        "startdatetime": inicio,
        "enddatetime":   fin,
    }
    try:
        respuesta = requests.get(GDELT_URL, params=params, timeout=30)
        respuesta.raise_for_status()
        datos = respuesta.json()
        return datos.get("articles", [])
    except requests.exceptions.Timeout:
        log.warning(f"Timeout en consulta {inicio[:6]} — se omite este mes")
        return []
    except requests.exceptions.HTTPError as e:
        log.warning(f"Error HTTP {e} en consulta {inicio[:6]} — se omite")
        return []
    except Exception as e:
        log.warning(f"Error inesperado en consulta {inicio[:6]}: {e}")
        return []


def extraer_empresa(ticker: str, query: str, meses: list[tuple]) -> pd.DataFrame:
    """
    Extrae todos los artículos de GDELT para una empresa iterando mes a mes.

    Parámetros
    ----------
    ticker : símbolo bursátil (p. ej. 'AAPL')
    query  : término de búsqueda para esta empresa
    meses  : lista de tuplas (inicio_mes, fin_mes) en formato YYYYMMDDHHMMSS

    Devuelve
    --------
    pd.DataFrame con todos los artículos en crudo.
    """
    registros = []
    nombre_empresa = query.split('"')[1]  # primer término entre comillas

    for inicio_mes, fin_mes in tqdm(
        meses,
        desc=f"  {ticker}",
        unit="mes",
        leave=False,
    ):
        mes_label = inicio_mes[:6]  # YYYYMM
        articulos = consultar_gdelt(query, inicio_mes, fin_mes)

        for art in articulos:
            tone_data = parsear_v2tone(art.get("V2Tone", ""))
            registros.append({
                # Campos en crudo de GDELT
                "seendate":       art.get("seendate"),
                "title":          art.get("title"),
                "url":            art.get("url"),
                "domain":         art.get("domain"),
                "language":       art.get("language"),
                "sourcecountry":  art.get("sourcecountry"),
                # Campos de sentimiento GDELT (V2Tone)
                "tone":           tone_data["tone"],
                "score_positivo": tone_data["score_positivo"],
                "score_negativo": tone_data["score_negativo"],
                "polaridad":      tone_data["polaridad"],
                "num_palabras":   tone_data["num_palabras"],
                # Metadatos de la consulta (trazabilidad)
                "query_ticker":   ticker,
                "query_empresa":  nombre_empresa,
                "query_mes":      f"{mes_label[:4]}-{mes_label[4:]}",
            })

        time.sleep(PAUSA_SEGUNDOS)

    if not registros:
        log.warning(f"{ticker}: no se encontraron artículos")
        return pd.DataFrame()

    df = pd.DataFrame(registros)

    # Eliminar duplicados exactos (mismo URL)
    n_antes = len(df)
    df = df.drop_duplicates(subset=["url"])
    n_dupes = n_antes - len(df)
    if n_dupes > 0:
        log.info(f"{ticker}: {n_dupes} duplicados eliminados")

    return df


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un DataFrame como CSV."""
    df.to_csv(ruta, index=False, encoding="utf-8")
    log.info(f"Guardado → {ruta.relative_to(RAIZ_PROYECTO)}  ({len(df):,} filas)")


def imprimir_resumen(resultados: dict[str, pd.DataFrame]) -> None:
    """Imprime resumen de la extracción de GDELT."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("  RESUMEN EXTRACCIÓN GDELT — Magnificent 7")
    print(sep)
    print(f"\n  {'Ticker':<8} {'Artículos':>10}  {'Tono medio':>11}  {'Tono std':>9}  {'Sin tono':>8}")
    print(f"  {'------':<8} {'----------':>10}  {'-----------':>11}  {'---------':>9}  {'--------':>8}")

    total = 0
    for ticker, df in resultados.items():
        if df.empty:
            print(f"  {ticker:<8} {'0':>10}  {'N/A':>11}  {'N/A':>9}  {'N/A':>8}")
            continue
        n = len(df)
        total += n
        tono = df["tone"].dropna()
        sin_tono = df["tone"].isna().sum()
        print(
            f"  {ticker:<8} {n:>10,}  "
            f"{tono.mean():>+11.4f}  "
            f"{tono.std():>9.4f}  "
            f"{sin_tono:>8}"
        )

    print(f"\n{sep}")
    print(f"  Total artículos extraídos: {total:,}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Inicio extracción de noticias GDELT ===")
    log.info(f"Periodo: {FECHA_INICIO.date()} → {FECHA_FIN.date()}")
    log.info(f"Empresas: {', '.join(EMPRESAS.keys())}")

    meses = generar_meses(FECHA_INICIO, FECHA_FIN)
    log.info(f"Consultas por empresa: {len(meses)} meses × {len(EMPRESAS)} empresas = {len(meses)*len(EMPRESAS)} llamadas API")

    resultados = {}
    frames = []

    for ticker, query in EMPRESAS.items():
        log.info(f"Extrayendo {ticker} ...")
        df = extraer_empresa(ticker, query, meses)
        resultados[ticker] = df

        if not df.empty:
            nombre_csv = DIR_SALIDA / f"{ticker}_gdelt_2019_2024.csv"
            guardar_csv(df, nombre_csv)
            frames.append(df)

    # CSV combinado
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        guardar_csv(df_all, DIR_SALIDA / "ALL_gdelt_2019_2024.csv")
        log.info(f"CSV combinado: {len(df_all):,} artículos en total")

    imprimir_resumen(resultados)
    log.info("=== Extracción GDELT completada ===")


if __name__ == "__main__":
    main()
