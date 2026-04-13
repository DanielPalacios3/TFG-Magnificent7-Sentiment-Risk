"""
Script: download_news.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
-----------------------
Se conecta a la API pública de GDELT Project (v2 Doc) y descarga artículos de noticias
financieras para cada una de las 7 empresas Magnificent 7, cubriendo el periodo completo
de 2019-01-01 a 2025-12-31. Lo importante aquí es que los datos se guardan TAL CUAL
vienen de la API, en crudo, sin limpiar ni filtrar nada. Ya se procesarán después en la
fase de limpieza y en el análisis NLP.

Como la API de GDELT solo devuelve un máximo de 250 artículos por consulta, la estrategia
es hacer una consulta por cada mes y por cada empresa. Así maximizamos la cobertura aunque
no capturemos absolutamente todo lo que existe en GDELT.

¿Qué archivos genera?
----------------------
- data/raw/news/{TICKER}_gdelt_2019_2025.csv  → un CSV individual por empresa
- data/raw/news/ALL_gdelt_2019_2025.csv       → un CSV combinado con todas las empresas juntas

¿Por qué usamos GDELT y no otra fuente de noticias?
----------------------------------------------------
GDELT (Global Database of Events, Language, and Tone) es una base de datos global de
noticias que tiene dos grandes ventajas para este TFG: es completamente gratuita (no
necesita API key ni plan de pago) y además ya calcula automáticamente el tono/sentimiento
de cada artículo con su propio sistema de análisis de texto. Eso nos da una señal de
sentimiento pre-calculada "de regalo" que luego podemos comparar con lo que saque FinBERT
de los textos de Reddit. Otras fuentes como Finnhub o Stock News API se investigaron pero
no cubrían el periodo histórico que necesitábamos.

Campos que se extraen de cada artículo (todos en crudo):
- seendate         : fecha y hora en que GDELT vio el artículo publicado
- title            : el titular de la noticia
- url              : enlace original al artículo
- domain           : el dominio del medio (reuters.com, bbc.co.uk, etc.)
- language         : idioma que GDELT detectó
- sourcecountry    : país de origen del medio de comunicación
- tone             : tono global del artículo, de -100 a +100 (primer valor del campo V2Tone)
- score_positivo   : densidad de palabras positivas en el texto (V2Tone[1])
- score_negativo   : densidad de palabras negativas en el texto (V2Tone[2])
- polaridad        : diferencia entre densidad positiva y negativa (V2Tone[3])
- num_palabras     : longitud del artículo en número de palabras (V2Tone[6])
- query_ticker     : el ticker por el que encontramos este artículo
- query_empresa    : el nombre de empresa que se usó en la consulta
- query_mes        : mes de la consulta en formato YYYY-MM (útil para trazabilidad)
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
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────

# Aquí definimos qué buscamos en GDELT para cada ticker. Usamos el nombre de la empresa
# entre comillas (búsqueda exacta) y filtramos solo artículos en inglés. Nota: probamos
# con operadores OR para capturar variantes (ej. "Alphabet" OR "Google") pero GDELT no
# los soporta bien, así que nos quedamos con el nombre principal de cada empresa.
EMPRESAS = {
    "AAPL":  '"Apple" sourcelang:english',
    "MSFT":  '"Microsoft" sourcelang:english',
    "GOOGL": '"Alphabet" sourcelang:english',   # idealmente buscaríamos "Alphabet" OR "Google" pero GDELT no lo soporta
    "AMZN":  '"Amazon" sourcelang:english',
    "NVDA":  '"NVIDIA" sourcelang:english',
    "META":  '"Meta" sourcelang:english',        # lo mismo: no podemos buscar "Meta" OR "Facebook"
    "TSLA":  '"Tesla" sourcelang:english',
}

# Si algún ticker ya lo descargamos en una ejecución anterior, lo metemos aquí para
# no repetir el trabajo. Vacío = descargar todo desde cero.
TICKERS_COMPLETADOS = set()

FECHA_INICIO  = datetime(2019, 1, 1)
FECHA_FIN     = datetime(2025, 12, 31)
PAUSA_SEGUNDOS = 12.0  # esperamos 12 segundos entre llamadas para no saturar la API pública y que nos bloquee (HTTP 429)
MAX_REGISTROS  = 250   # GDELT solo devuelve un máximo de 250 artículos por consulta, no se puede subir

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
    Genera una lista de tuplas (fecha_inicio, fecha_fin) para cada mes dentro
    del rango dado. Las fechas vienen en el formato que espera la API de GDELT:
    YYYYMMDDHHMMSS. Esto nos permite luego iterar mes a mes haciendo una consulta
    por cada uno, que es la forma de maximizar la cobertura con el límite de 250
    artículos por consulta.
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
    El campo V2Tone de GDELT viene como un string con 7 valores separados por comas.
    Cada posición tiene un significado distinto:
      [0] tono global (-100 a +100, donde negativo = pesimista y positivo = optimista)
      [1] densidad de palabras positivas en el texto
      [2] densidad de palabras negativas
      [3] polaridad (positivas - negativas)
      [4] densidad de referencias a actividad (no la usamos)
      [5] densidad de auto-referencia (tampoco la usamos)
      [6] número total de palabras del artículo

    Esta función extrae los campos que nos interesan para el TFG y los devuelve
    como un diccionario. Si el string viene mal formado, devuelve todo como None
    para no romper la ejecución.
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
    Hace una llamada a la API de GDELT v2 Doc con los parámetros dados y devuelve
    la lista de artículos que encuentre. Si algo falla (timeout, error HTTP, lo que sea),
    simplemente logea un warning y devuelve una lista vacía para que el script siga
    adelante sin pararse. Más vale perder un mes de datos que cascar toda la extracción.

    Parámetros
    ----------
    query  : lo que queremos buscar, en el formato que espera GDELT (ej. '"Apple" sourcelang:english')
    inicio : desde cuándo buscar, en formato YYYYMMDDHHMMSS
    fin    : hasta cuándo buscar, en el mismo formato

    Devuelve
    --------
    Lista de dicts con los campos de cada artículo encontrado, o lista vacía si hubo algún problema.
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
    Esta es la función principal de extracción: para una empresa dada, recorre todos
    los meses del periodo y va haciendo consultas a GDELT una por una, acumulando
    todos los artículos que encuentra. Entre consulta y consulta espera unos segundos
    para no saturar la API. Al final, elimina duplicados (por URL) y devuelve todo
    como un DataFrame en crudo.

    Parámetros
    ----------
    ticker : el símbolo bursátil de la empresa (ej. 'AAPL', 'TSLA')
    query  : el texto de búsqueda para GDELT (ej. '"Apple" sourcelang:english')
    meses  : lista de tuplas (inicio_mes, fin_mes) ya en formato YYYYMMDDHHMMSS

    Devuelve
    --------
    pd.DataFrame con todos los artículos encontrados, sin limpiar, listos para guardar.
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
                # Campos que vienen directamente de la respuesta de GDELT
                "seendate":       art.get("seendate"),
                "title":          art.get("title"),
                "url":            art.get("url"),
                "domain":         art.get("domain"),
                "language":       art.get("language"),
                "sourcecountry":  art.get("sourcecountry"),
                # Campos de sentimiento que extraemos del V2Tone (ya parseados)
                "tone":           tone_data["tone"],
                "score_positivo": tone_data["score_positivo"],
                "score_negativo": tone_data["score_negativo"],
                "polaridad":      tone_data["polaridad"],
                "num_palabras":   tone_data["num_palabras"],
                # Metadatos que añadimos nosotros para poder rastrear de dónde salió cada artículo
                "query_ticker":   ticker,
                "query_empresa":  nombre_empresa,
                "query_mes":      f"{mes_label[:4]}-{mes_label[4:]}",
            })

        time.sleep(PAUSA_SEGUNDOS)

    if not registros:
        log.warning(f"{ticker}: no se encontraron artículos")
        return pd.DataFrame()

    df = pd.DataFrame(registros)

    # Quitamos artículos duplicados (misma URL = mismo artículo contado dos veces)
    n_antes = len(df)
    df = df.drop_duplicates(subset=["url"])
    n_dupes = n_antes - len(df)
    if n_dupes > 0:
        log.info(f"{ticker}: {n_dupes} duplicados eliminados")

    return df


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un DataFrame como CSV en la ruta indicada y logea cuántas filas tiene."""
    df.to_csv(ruta, index=False, encoding="utf-8")
    log.info(f"Guardado → {ruta.relative_to(RAIZ_PROYECTO)}  ({len(df):,} filas)")


def imprimir_resumen(resultados: dict[str, pd.DataFrame]) -> None:
    """Muestra una tabla resumen al final de la extracción con el número de artículos por
    empresa, el tono medio, la desviación estándar del tono y cuántos artículos no tenían
    tono calculado. Así se ve de un vistazo si la extracción fue bien o si algún ticker
    se quedó vacío."""
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
# MAIN — Aquí se orquesta todo: genera los meses, itera por empresa, descarga
# los artículos, guarda CSVs individuales y al final combina todo en uno solo.
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
        if ticker in TICKERS_COMPLETADOS:
            log.info(f"Saltando {ticker} (ya descargado)")
            continue
        log.info(f"Extrayendo {ticker} ...")
        df = extraer_empresa(ticker, query, meses)
        resultados[ticker] = df

        if not df.empty:
            nombre_csv = DIR_SALIDA / f"{ticker}_gdelt_2019_2025.csv"
            guardar_csv(df, nombre_csv)
            frames.append(df)

    # Al final juntamos todo en un solo CSV maestro (incluyendo los de ejecuciones anteriores)
    csvs_existentes = list(DIR_SALIDA.glob("*_gdelt_2019_2025.csv"))
    csvs_existentes = [f for f in csvs_existentes if not f.name.startswith("ALL_")]
    todos = [pd.read_csv(f) for f in csvs_existentes] + frames
    if todos:
        df_all = pd.concat(todos, ignore_index=True)
        guardar_csv(df_all, DIR_SALIDA / "ALL_gdelt_2019_2025.csv")
        log.info(f"CSV combinado: {len(df_all):,} artículos en total")

    imprimir_resumen(resultados)
    log.info("=== Extracción GDELT completada ===")


if __name__ == "__main__":
    main()
