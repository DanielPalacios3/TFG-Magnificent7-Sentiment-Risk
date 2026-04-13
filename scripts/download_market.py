"""
Script: download_market.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Este es el primer paso de todo el pipeline de datos. Se conecta a Yahoo Finance y se
descarga los precios de apertura, cierre, máximo, mínimo y volumen (lo que se conoce
como datos OHLCV) para cada una de las 7 empresas del grupo Magnificent 7. El rango
temporal va desde el 1 de enero de 2019 hasta el 5 de abril de 2026, que es todo el
periodo que necesitamos para el TFG. Después de descargar los precios en crudo, el
script calcula una serie de variables derivadas que van a ser el input de los modelos
GARCH y XGBoost, y lo guarda todo en la carpeta data/raw/market/.

¿Qué archivos genera?
---------------------
- data/raw/market/{TICKER}_market_2019_2026.csv  → un CSV individual por cada empresa
- data/raw/market/ALL_market_2019_2026.csv       → un CSV grande con las 7 juntas

¿Y por qué calculamos cada variable? Aquí va la explicación de cada una:
------------------------------------------------------------------------
- log_return         : es el rendimiento logarítmico diario, o sea, cuánto sube o baja
                       la acción cada día en escala log. Es el input principal tanto para
                       GARCH como para XGBoost. Usamos logaritmos (y no rendimientos simples)
                       porque tienen propiedades estadísticas muy convenientes: se pueden
                       sumar entre días y su distribución se parece más a una normal.
- vol_realized_5d    : la volatilidad realizada con una ventana rolling de 5 días, anualizada.
                       Básicamente captura cuánto se ha movido la acción en la última semana
                       de trading. Es útil como variable rezagada en el modelo GARCH-X para
                       ver si la volatilidad reciente ayuda a predecir la futura.
- vol_realized_20d   : la volatilidad realizada con ventana de 20 días (anualizada), que
                       equivale aproximadamente a un mes de negociación. Esta es la VARIABLE
                       OBJETIVO principal del TFG — lo que los modelos intentan predecir.
- vol_intraday       : un proxy de la volatilidad dentro del día, calculado como
                       (High - Low) / Close. Es una versión simplificada del rango de
                       Parkinson, y la gracia es que aprovecha que tenemos precios OHLC
                       (no solo el cierre) para tener una medida de volatilidad más rica.
- abs_return         : el valor absoluto del rendimiento diario, que es una forma directa
                       y sencilla de medir volatilidad. Es un clásico en la literatura
                       financiera (Ding et al., 1993 ya lo usaban).
- sq_return          : el rendimiento al cuadrado, que es esencialmente un proxy de la
                       varianza diaria. Es la base teórica del modelo ARCH original de
                       Engle (1982), y por tanto de todo el marco GARCH que usamos.
- log_volume_change  : el cambio logarítmico en el volumen de negociación de un día a otro.
                       Lo incluimos como candidata para GARCH-X porque el volumen puede ser
                       un proxy interesante de la atención y actividad del mercado.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — aquí se definen los parámetros principales del script
# ─────────────────────────────────────────────────────────────────────────────

TICKERS      = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
FECHA_INICIO = "2019-01-01"
FECHA_FIN    = "2026-04-05"
FACTOR_ANUAL = np.sqrt(252)  # para pasar de volatilidad diaria a anualizada (252 días hábiles al año)

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
# FUNCIONES — cada una se encarga de un paso concreto del proceso
# ─────────────────────────────────────────────────────────────────────────────

def descargar_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Se conecta a Yahoo Finance y descarga los precios diarios (apertura, máximo,
    mínimo, cierre y volumen) para una empresa concreta. Es la función base que
    alimenta todo el resto del script.

    Parámetros
    ----------
    ticker : str  — el símbolo bursátil de la empresa, por ejemplo 'AAPL' para Apple.

    Devuelve
    --------
    Un DataFrame de pandas con las columnas open, high, low, close y volume,
    indexado por fecha y ordenado cronológicamente.
    """
    datos = yf.download(
        ticker,
        start=FECHA_INICIO,
        end=FECHA_FIN,
        auto_adjust=True,  # esto hace que yfinance ajuste automáticamente los splits y dividendos
        progress=False,
    )

    if datos.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    # A veces yfinance devuelve columnas con MultiIndex, así que lo aplanamos por si acaso
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    datos.columns = [c.lower() for c in datos.columns]
    datos.index.name = "date"
    return datos[["open", "high", "low", "close", "volume"]].sort_index()


def calcular_variables(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    A partir de los precios en crudo, calcula todas las variables derivadas que
    necesitamos para los modelos del TFG. Esto incluye rendimientos, volatilidades
    rolling, proxies de volatilidad intradía y cambios en volumen. Básicamente
    transforma el DataFrame de precios simples en algo listo para modelar.

    Parámetros
    ----------
    df     : DataFrame con las columnas OHLCV originales (el índice tiene que ser la fecha).
    ticker : el símbolo bursátil de la empresa, que se mete como columna para poder
             identificar cada fila cuando luego juntemos todo en un solo CSV.

    Devuelve
    --------
    El mismo DataFrame pero enriquecido con todas las variables calculadas.
    """
    df = df.copy()

    # Rendimiento logarítmico diario: log(cierre_hoy / cierre_ayer)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatilidad realizada con ventana rolling, anualizada multiplicando por sqrt(252)
    df["vol_realized_5d"]  = df["log_return"].rolling(5).std()  * FACTOR_ANUAL
    df["vol_realized_20d"] = df["log_return"].rolling(20).std() * FACTOR_ANUAL

    # Proxy de volatilidad intradía usando el rango del día (versión simplificada del rango de Parkinson)
    df["vol_intraday"] = (df["high"] - df["low"]) / df["close"]

    # Estas dos son proxies clásicos de volatilidad: el absoluto y el cuadrado del rendimiento
    df["abs_return"] = df["log_return"].abs()
    df["sq_return"]  = df["log_return"] ** 2

    # Cambio logarítmico en volumen: si el volumen sube mucho puede indicar que algo gordo está pasando
    df["log_volume_change"] = np.log(df["volume"] / df["volume"].shift(1))

    # Metemos el ticker como primera columna para saber de qué empresa es cada fila
    df.insert(0, "ticker", ticker)

    return df


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un DataFrame en formato CSV, manteniendo la fecha como índice."""
    df.to_csv(ruta, index=True, date_format="%Y-%m-%d")
    log.info(f"Guardado → {ruta.relative_to(RAIZ_PROYECTO)}")


def imprimir_resumen(df_all: pd.DataFrame) -> None:
    """Imprime un resumen completo de lo que se ha descargado: cobertura temporal,
    estadísticas de rendimientos, volatilidad media y valores nulos por empresa."""
    sep = "─" * 74

    print(f"\n{sep}")
    print("  RESUMEN DE DATOS DESCARGADOS — Magnificent 7")
    print(sep)

    grupos = df_all.groupby("ticker")

    # Primero mostramos cuántas filas tiene cada empresa y de qué fecha a qué fecha
    print("\n  COBERTURA TEMPORAL\n")
    print(f"  {'Ticker':<8} {'Filas':>6}  {'Inicio':>12}  {'Fin':>12}")
    print(f"  {'------':<8} {'------':>6}  {'----------':>12}  {'----------':>12}")
    for ticker, g in grupos:
        print(f"  {ticker:<8} {len(g):>6}  {str(g.index.min().date()):>12}  {str(g.index.max().date()):>12}")

    # Estadísticas básicas de los rendimientos logarítmicos
    print("\n  RENDIMIENTOS LOGARÍTMICOS DIARIOS\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        r = g["log_return"].dropna()
        print(f"  {ticker:<8} {r.mean():>10.6f}  {r.std():>10.6f}  {r.min():>10.6f}  {r.max():>10.6f}")

    # Volatilidad realizada con ventana de 20 días (nuestra variable objetivo)
    print("\n  VOLATILIDAD REALIZADA MEDIA (ventana 20d, anualizada)\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        v = g["vol_realized_20d"].dropna()
        print(f"  {ticker:<8} {v.mean():>10.4f}  {v.std():>10.4f}  {v.max():>10.4f}")

    # Revisamos si hay nulos (los primeros registros rolling siempre son NaN, eso es normal)
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
# MAIN — aquí se orquesta todo: descarga, cálculo de variables y guardado
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Inicio descarga de datos de mercado ===")
    log.info(f"Periodo: {FECHA_INICIO} → {FECHA_FIN} | Empresas: {', '.join(TICKERS)}")

    frames = []

    for ticker in tqdm(TICKERS, desc="Descargando", unit="ticker"):
        try:
            df_raw = descargar_ohlcv(ticker)
            df     = calcular_variables(df_raw, ticker)
            guardar_csv(df, DIR_SALIDA / f"{ticker}_market_2019_2026.csv")
            frames.append(df)
        except Exception as e:
            log.error(f"Error con {ticker}: {e}")
            sys.exit(1)

    # Juntamos todos los DataFrames en un solo CSV combinado con las 7 empresas
    df_all = pd.concat(frames).sort_values(["ticker", "date"])
    guardar_csv(df_all, DIR_SALIDA / "ALL_market_2019_2026.csv")
    log.info(f"CSV combinado: {len(df_all):,} filas en total")

    imprimir_resumen(df_all)
    log.info("=== Descarga completada ===")


if __name__ == "__main__":
    main()
