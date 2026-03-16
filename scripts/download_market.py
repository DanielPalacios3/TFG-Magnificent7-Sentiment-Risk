"""
Script: download_market.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace?
----------
Descarga precios OHLCV diarios de las 7 empresas Magnificent 7 desde Yahoo Finance
para el periodo 2019-01-01 a 2024-12-31, calcula variables de mercado relevantes
para el TFG y guarda los datos en data/raw/market/.

¿Qué genera?
------------
- data/raw/market/{TICKER}_market_2019_2024.csv  → un CSV por empresa
- data/raw/market/ALL_market_2019_2024.csv       → CSV combinado con las 7 empresas

¿Por qué cada variable?
-----------------------
- log_return         : rendimiento logarítmico diario — input principal de los modelos
                       GARCH y XGBoost. Se usa logarítmico por sus propiedades estadísticas
                       (aditividad temporal, distribución más cercana a normal).
- vol_realized_5d    : volatilidad realizada rolling 5 días (anualizada). Captura el riesgo
                       a muy corto plazo; útil como variable rezagada en GARCH-X.
- vol_realized_20d   : volatilidad realizada rolling 20 días (anualizada). Aproxima un mes
                       de negociación; es la VARIABLE OBJETIVO principal del TFG.
- vol_intraday       : proxy de volatilidad intradía = (High - Low) / Close (rango de Parkinson
                       simplificado). Aprovecha precios OHLC para una medida de volatilidad
                       más precisa que usar solo el precio de cierre.
- abs_return         : valor absoluto del rendimiento — proxy directo de volatilidad diaria,
                       habitual en la literatura (Ding et al., 1993).
- sq_return          : rendimiento al cuadrado — proxy de varianza diaria, base del modelo
                       ARCH (Engle, 1982) y por tanto de todo el marco GARCH.
- log_volume_change  : cambio logarítmico en volumen — variable candidata a incluir en
                       GARCH-X como proxy de actividad y atención del mercado.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

TICKERS      = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
FECHA_INICIO = "2019-01-01"
FECHA_FIN    = "2024-12-31"
FACTOR_ANUAL = np.sqrt(252)  # factor de anualización para volatilidad diaria

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_SALIDA    = RAIZ_PROYECTO / "data" / "raw" / "market"
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

def descargar_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Descarga precios OHLCV diarios de Yahoo Finance para un ticker dado.

    Parámetros
    ----------
    ticker : str  — símbolo bursátil (p. ej. 'AAPL').

    Devuelve
    --------
    pd.DataFrame con columnas: open, high, low, close, volume.
    """
    datos = yf.download(
        ticker,
        start=FECHA_INICIO,
        end=FECHA_FIN,
        auto_adjust=True,  # ajusta splits y dividendos automáticamente
        progress=False,
    )

    if datos.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    # Aplanar MultiIndex si yfinance lo devuelve así
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    datos.columns = [c.lower() for c in datos.columns]
    datos.index.name = "date"
    return datos[["open", "high", "low", "close", "volume"]].sort_index()


def calcular_variables(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calcula las variables de mercado derivadas necesarias para el TFG.

    Parámetros
    ----------
    df     : DataFrame con columnas OHLCV (índice = fecha).
    ticker : símbolo bursátil; se añade como columna identificadora.

    Devuelve
    --------
    pd.DataFrame enriquecido con todas las variables calculadas.
    """
    df = df.copy()

    # Rendimiento logarítmico diario
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatilidad realizada rolling (anualizada)
    df["vol_realized_5d"]  = df["log_return"].rolling(5).std()  * FACTOR_ANUAL
    df["vol_realized_20d"] = df["log_return"].rolling(20).std() * FACTOR_ANUAL

    # Proxy de volatilidad intradía (rango de Parkinson simplificado)
    df["vol_intraday"] = (df["high"] - df["low"]) / df["close"]

    # Rendimiento absoluto y al cuadrado
    df["abs_return"] = df["log_return"].abs()
    df["sq_return"]  = df["log_return"] ** 2

    # Cambio logarítmico en volumen
    df["log_volume_change"] = np.log(df["volume"] / df["volume"].shift(1))

    # Identificador de empresa (primera columna)
    df.insert(0, "ticker", ticker)

    return df


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un DataFrame como CSV con índice de fecha."""
    df.to_csv(ruta, index=True, date_format="%Y-%m-%d")
    log.info(f"Guardado → {ruta.relative_to(RAIZ_PROYECTO)}")


def imprimir_resumen(df_all: pd.DataFrame) -> None:
    """Imprime resumen estadístico de los datos descargados por empresa."""
    sep = "─" * 74

    print(f"\n{sep}")
    print("  RESUMEN DE DATOS DESCARGADOS — Magnificent 7")
    print(sep)

    grupos = df_all.groupby("ticker")

    # Cobertura temporal
    print("\n  COBERTURA TEMPORAL\n")
    print(f"  {'Ticker':<8} {'Filas':>6}  {'Inicio':>12}  {'Fin':>12}")
    print(f"  {'------':<8} {'------':>6}  {'----------':>12}  {'----------':>12}")
    for ticker, g in grupos:
        print(f"  {ticker:<8} {len(g):>6}  {str(g.index.min().date()):>12}  {str(g.index.max().date()):>12}")

    # Rendimientos
    print("\n  RENDIMIENTOS LOGARÍTMICOS DIARIOS\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        r = g["log_return"].dropna()
        print(f"  {ticker:<8} {r.mean():>10.6f}  {r.std():>10.6f}  {r.min():>10.6f}  {r.max():>10.6f}")

    # Volatilidad realizada 20d
    print("\n  VOLATILIDAD REALIZADA MEDIA (ventana 20d, anualizada)\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        v = g["vol_realized_20d"].dropna()
        print(f"  {ticker:<8} {v.mean():>10.4f}  {v.std():>10.4f}  {v.max():>10.4f}")

    # Nulos
    print("\n  VALORES NULOS POR COLUMNA\n")
    nulos = df_all.isnull().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print("  Ninguno inesperado (los primeros registros rolling son NaN por diseño).")
    else:
        for col, n in nulos.items():
            pct = 100 * n / len(df_all)
            print(f"  {col:<25} {n:>6} nulos  ({pct:.1f}%)")

    print(f"\n{sep}")
    print(f"  Total registros: {len(df_all):,}  |  Empresas: {df_all['ticker'].nunique()}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Inicio descarga de datos de mercado ===")
    log.info(f"Periodo: {FECHA_INICIO} → {FECHA_FIN} | Empresas: {', '.join(TICKERS)}")

    frames = []

    for ticker in tqdm(TICKERS, desc="Descargando", unit="ticker"):
        try:
            df_raw = descargar_ohlcv(ticker)
            df     = calcular_variables(df_raw, ticker)
            guardar_csv(df, DIR_SALIDA / f"{ticker}_market_2019_2024.csv")
            frames.append(df)
        except Exception as e:
            log.error(f"Error con {ticker}: {e}")
            sys.exit(1)

    # CSV combinado
    df_all = pd.concat(frames).sort_values(["ticker", "date"])
    guardar_csv(df_all, DIR_SALIDA / "ALL_market_2019_2024.csv")
    log.info(f"CSV combinado: {len(df_all):,} filas en total")

    imprimir_resumen(df_all)
    log.info("=== Descarga completada ===")


if __name__ == "__main__":
    main()
