"""
Script: download_news_bigquery_update_2026.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Es la actualización incremental de noticias GDELT. Descarga los artículos de
enero a marzo de 2026 desde BigQuery, que son los 3 meses que nos faltaban para
que la cobertura de noticias llegue hasta abril 2026 y quede alineada con los
datos intradía de mercado. Toma los artículos nuevos y los appende a los CSVs
que ya teníamos, deduplicando por URL para no meter repetidos.

¿Por qué seguimos usando BigQuery?
------------------------------------
Los CSVs existentes (2019-2025) se descargaron con BigQuery y tienen un formato
concreto (fecha, url, domain, tone, organizations, persons, locations, etc.).
Para mantener la consistencia usamos exactamente la misma fuente y la misma
consulta SQL, solo cambiando el rango de fechas.

¿Qué archivos modifica?
-----------------------
Actualiza los CSVs existentes y crea versiones con nombre extendido:
- data/raw/news/{TICKER}_gdelt_2019_2026.csv  (con los datos nuevos appendeados)
- data/raw/news/{TICKER}_gdelt_2019_2025.csv  (también se actualiza para compatibilidad)

Autenticación necesaria
-----------------------
    gcloud auth application-default login

Uso
---
    source venv/bin/activate
    python scripts/download_news_bigquery_update_2026.py
"""

import os
from datetime import date

import pandas as pd
from google.cloud import bigquery

# ── Configuración — mismos parámetros que el script original, pero solo Q1 2026 ─

GCP_PROJECT = "tfg-magnificent7"

# Solo descargamos el periodo que nos faltaba: primer trimestre de 2026
FECHA_INICIO = "2026-01-01"
FECHA_FIN    = "2026-03-31"

EMPRESAS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
}

DIR_NEWS = os.path.join("data", "raw", "news")

# ── Consulta — la misma SQL que el script original, sin cambios ──────────────

QUERY_TEMPLATE = """
SELECT
    DATE(_PARTITIONTIME)          AS fecha,
    DocumentIdentifier            AS url,
    SourceCommonName              AS domain,
    SPLIT(V2Tone, ',')[OFFSET(0)] AS tone,
    Organizations                 AS organizations,
    Persons                       AS persons,
    Locations                     AS locations,
    '{ticker}'                    AS query_ticker,
    '{empresa}'                   AS query_empresa
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('{fecha_inicio}')
                       AND TIMESTAMP('{fecha_fin} 23:59:59')
    AND (
        LOWER(Organizations) LIKE LOWER('%{empresa}%')
        OR LOWER(Persons)    LIKE LOWER('%{empresa}%')
    )
    AND V2Tone IS NOT NULL
    AND V2Tone != ''
ORDER BY fecha
"""


def descargar_y_append(cliente, ticker, empresa):
    """Descarga los artículos de enero a marzo de 2026 para un ticker concreto
    y los añade al CSV existente. Deduplica por URL para evitar repetidos.
    Devuelve una tupla con (total_filas_final, filas_nuevas_añadidas)."""
    # Las dos rutas donde puede estar el CSV (nombre viejo y nuevo)
    ruta_existente = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2025.csv")
    ruta_nueva     = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2026.csv")

    # Cargamos el CSV que ya teníamos para tener las URLs existentes y poder deduplicar
    if os.path.exists(ruta_existente):
        df_old = pd.read_csv(ruta_existente)
        urls_existentes = set(df_old['url'].dropna())
        n_antes = len(df_old)
    else:
        df_old = pd.DataFrame()
        urls_existentes = set()
        n_antes = 0

    print(f"  [{ticker}] Consultando BigQuery para '{empresa}' ({FECHA_INICIO} → {FECHA_FIN})...")

    query = QUERY_TEMPLATE.format(
        ticker=ticker, empresa=empresa,
        fecha_inicio=FECHA_INICIO, fecha_fin=FECHA_FIN,
    )

    try:
        job_config = bigquery.QueryJobConfig(use_query_cache=True, use_legacy_sql=False)
        df_new = cliente.query(query, job_config=job_config).to_dataframe()
    except Exception as e:
        print(f"  [{ticker}] ERROR BigQuery: {e}")
        return 0, 0

    if df_new.empty:
        print(f"  [{ticker}] Sin resultados para ene-mar 2026")
        return n_antes, 0

    # BigQuery devuelve la fecha como objeto datetime.date, la pasamos a string para consistencia
    df_new['fecha'] = df_new['fecha'].astype(str)

    # Quitamos los artículos que ya teníamos (deduplicamos por URL)
    df_new = df_new[~df_new['url'].isin(urls_existentes)]
    n_nuevos = len(df_new)

    if n_nuevos == 0:
        print(f"  [{ticker}] Todos los artículos ya existían")
        return n_antes, 0

    # Juntamos los datos nuevos con los existentes y guardamos en ambos nombres de archivo
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final = df_final.sort_values('fecha').reset_index(drop=True)

    df_final.to_csv(ruta_nueva, index=False, encoding="utf-8")
    df_final.to_csv(ruta_existente, index=False, encoding="utf-8")

    n_despues = len(df_final)
    print(f"  [{ticker}] {n_antes:,} → {n_despues:,} filas (+{n_nuevos:,} nuevas)")
    print(f"  Guardado: {ruta_nueva}")

    return n_despues, n_nuevos


def main():
    os.makedirs(DIR_NEWS, exist_ok=True)

    print("=" * 60)
    print("  GDELT BigQuery Update — ene-mar 2026")
    print(f"  Proyecto GCP : {GCP_PROJECT}")
    print(f"  Periodo      : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  Tickers      : {', '.join(EMPRESAS.keys())}")
    print("=" * 60)

    cliente = bigquery.Client(project=GCP_PROJECT)

    resumen = {}
    for ticker, empresa in EMPRESAS.items():
        n_total, n_nuevos = descargar_y_append(cliente, ticker, empresa)
        resumen[ticker] = (n_total, n_nuevos)

    # Generamos el CSV combinado juntando todos los tickers
    print(f"\n  Generando CSV combinado...")
    frames = []
    for ticker in EMPRESAS:
        ruta = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2026.csv")
        if not os.path.exists(ruta):
            ruta = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2025.csv")
        if os.path.exists(ruta):
            frames.append(pd.read_csv(ruta))
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all.to_csv(os.path.join(DIR_NEWS, "ALL_gdelt_2019_2026.csv"),
                      index=False, encoding="utf-8")
        print(f"  ALL_gdelt_2019_2026.csv: {len(df_all):,} filas totales")

    # Resumen final: cuántos artículos nuevos se añadieron por cada empresa
    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    total_nuevos = 0
    for ticker, (n_total, n_nuevos) in resumen.items():
        print(f"  {ticker:5s}: {n_total:>10,} filas  (+{n_nuevos:,} nuevas)")
        total_nuevos += n_nuevos
    print(f"\n  Total artículos nuevos: {total_nuevos:,}")
    print(f"  Actualización completada.")


if __name__ == "__main__":
    main()
