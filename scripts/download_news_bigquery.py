"""
Script: download_news_bigquery.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Este script se encarga de descargar artículos de noticias financieras históricas
relacionadas con las Magnificent 7 desde la base de datos GDELT, utilizando Google
BigQuery como motor de consultas. La tabla que consultamos es la particionada
gdelt-bq.gdeltv2.gkg_partitioned. De cada artículo extraemos el tono (V2Tone),
que es un score de sentimiento pre-calculado por GDELT, y que luego usaremos como
variable exógena en los modelos GARCH-X y como feature en XGBoost.

¿Por qué usamos BigQuery en vez de la API REST de GDELT?
---------------------------------------------------------
Investigamos la API REST de GDELT DOC, pero resulta que solo da acceso a los últimos
3 meses de datos. Como nuestro periodo de estudio es 2019-2025 (necesitamos histórico
largo), no nos servía. BigQuery en cambio tiene cobertura desde 2015 y se puede
consultar gratis dentro del free tier de Google Cloud (1 TB al mes de consultas, que
nos sobra de largo para este proyecto).

¿Y por qué el tono de GDELT y no otro indicador de sentimiento?
----------------------------------------------------------------
GDELT calcula automáticamente un campo llamado V2Tone para cada artículo que indexa.
Es un score numérico donde los valores altos son positivos y los bajos negativos.
La ventaja es que ya viene calculado, no hace falta aplicar NLP propio. No es tan
sofisticado como FinBERT (que sí usamos con Reddit), pero cubre millones de artículos
desde 2019 sin coste adicional. En la práctica funciona como un léxico de sentimiento
a gran escala.

¿Qué archivos genera?
---------------------
- data/raw/news/{TICKER}_gdelt_2019_2025.csv  → un CSV por cada empresa
- Cada CSV tiene estas columnas: fecha, title, url, domain, tone, organizations,
  persons, country, query_ticker, query_empresa

Autenticación necesaria
-----------------------
Antes de ejecutar hay que tener las credenciales de Google Cloud activadas:
    gcloud auth application-default login
El proyecto de GCP configurado es TFG-Magnificent7 (o el que tengas activo en gcloud).

Uso
---
    source venv/bin/activate
    python scripts/download_news_bigquery.py
"""

import os
from datetime import date

from google.cloud import bigquery

# ── Configuración — parámetros del proyecto y la consulta ────────────────────

# El proyecto de Google Cloud donde se facturan las consultas (tiene que coincidir
# con el que tengas autenticado en gcloud)
GCP_PROJECT = "tfg-magnificent7"

# Periodo de estudio completo — desde 2019 para tener histórico largo para GARCH
FECHA_INICIO = "2019-01-01"
FECHA_FIN = "2025-12-31"

# Las 7 empresas: mapeamos cada ticker al nombre que buscamos en GDELT
EMPRESAS = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "NVIDIA",
    "META":  "Meta",
    "TSLA":  "Tesla",
}

# Donde se guardan los CSVs resultantes
DIR_NEWS = os.path.join("data", "raw", "news")

# Si quieres limitar el número de filas por ticker, cambia esto (None = sin límite).
# En la práctica no hace falta porque cada consulta procesa entre 5 y 20 GB,
# y el free tier de BigQuery da 1 TB al mes de sobra.
MAX_FILAS = None

# ── Consulta BigQuery — la SQL que se ejecuta para cada empresa ──────────────

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
{limit_clause}
"""


def construir_query(ticker: str, empresa: str) -> str:
    """Monta la consulta SQL sustituyendo los placeholders por el ticker y empresa
    que toque. Si hay límite de filas configurado, lo añade al final de la query."""
    limit_clause = f"LIMIT {MAX_FILAS}" if MAX_FILAS else ""
    return QUERY_TEMPLATE.format(
        ticker=ticker,
        empresa=empresa,
        fecha_inicio=FECHA_INICIO,
        fecha_fin=FECHA_FIN,
        limit_clause=limit_clause,
    )


def descargar_ticker(cliente: bigquery.Client, ticker: str, empresa: str) -> int:
    """Lanza la consulta en BigQuery para un ticker concreto, espera a que termine
    y guarda el resultado como CSV. Si el archivo ya existe, se lo salta para no
    repetir la descarga (útil cuando se interrumpe el script y hay que relanzarlo).

    Devuelve el número de filas que se descargaron (o que ya existían).
    """
    ruta_csv = os.path.join(DIR_NEWS, f"{ticker}_gdelt_2019_2025.csv")

    # Si el CSV ya existe, nos lo saltamos para no gastar cuota de BigQuery innecesariamente
    if os.path.exists(ruta_csv):
        import pandas as pd
        filas = len(pd.read_csv(ruta_csv))
        print(f"  [{ticker}] Ya existe ({filas:,d} filas). Saltando.")
        return filas

    print(f"  [{ticker}] Consultando BigQuery para '{empresa}' ({FECHA_INICIO} → {FECHA_FIN})...")

    query = construir_query(ticker, empresa)

    try:
        # Configuración del job de BigQuery: usamos caché para no repetir consultas idénticas
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            use_legacy_sql=False,
        )

        df = cliente.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            print(f"  [{ticker}] Sin resultados. Revisar filtros.")
            return 0

        df.to_csv(ruta_csv, index=False, encoding="utf-8")
        print(f"  [{ticker}] {len(df):,d} filas guardadas → {ruta_csv}")
        return len(df)

    except Exception as e:
        print(f"  [{ticker}] ERROR: {e}")
        return 0


# ── Ejecución principal — aquí se lanza la descarga de todos los tickers ─────

def main():
    os.makedirs(DIR_NEWS, exist_ok=True)

    print("=" * 60)
    print("  GDELT BigQuery Downloader — TFG Magnificent 7")
    print(f"  Proyecto GCP : {GCP_PROJECT}")
    print(f"  Periodo      : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  Tickers      : {', '.join(EMPRESAS.keys())}")
    print("=" * 60)

    # Creamos el cliente de BigQuery apuntando a nuestro proyecto de GCP
    cliente = bigquery.Client(project=GCP_PROJECT)

    total_filas = 0
    for ticker, empresa in EMPRESAS.items():
        filas = descargar_ticker(cliente, ticker, empresa)
        total_filas += filas

    print("\n" + "=" * 60)
    print(f"  Descarga completada. Total filas: {total_filas:,d}")
    print(f"  Archivos en: {DIR_NEWS}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
