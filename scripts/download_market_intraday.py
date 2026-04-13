"""
Script: download_market_intraday.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

¿Qué hace este script?
----------------------
Este script descarga los precios de apertura, máximo, mínimo, cierre y volumen
(OHLCV) con granularidad horaria (cada barra = 1 hora) para las 7 empresas del
Magnificent 7 desde Yahoo Finance. El periodo que cubre es el máximo disponible
gratis, que son unos 2 años (aproximadamente de abril 2024 a abril 2026). A
partir de esos precios calcula un montón de variables adaptadas a escala horaria,
y lo guarda todo en data/raw/market_intraday/.

¿Y por qué necesitamos datos horarios si ya tenemos diarios?
-------------------------------------------------------------
Los datos diarios (2019-2025) están en data/raw/market/ y cubren un periodo mucho
más largo, pero con menos detalle. La idea de este script es generar una segunda
base de datos con granularidad horaria para ver si los modelos predictivos de
volatilidad mejoran cuando les das información más fina. La limitación es que
yfinance solo da ~730 días en datos de 1h, así que la ventana queda en 2024-2026.
Pero eso coincide justo con el periodo donde tenemos datos de Reddit y GDELT,
así que podemos hacer un estudio multifuente completo sin problemas.

¿Qué archivos genera?
---------------------
- data/raw/market_intraday/{TICKER}_market_intraday_2024_2026.csv  → uno por empresa
- data/raw/market_intraday/ALL_market_intraday_2024_2026.csv       → todas juntas

Explicación de cada variable que calculamos:
--------------------------------------------
- log_return_1h     : rendimiento logarítmico por barra horaria. Es lo mismo que el
                      log_return diario pero con unidad temporal de 1 hora.
- vol_realized_6h   : volatilidad realizada rolling de 6 barras, que equivale más o
                      menos a 1 día de negociación (el mercado abre 6.5 horas al día,
                      redondeamos a 6). Es el equivalente horario del vol_realized_5d
                      de la base diaria.
- vol_realized_30h  : volatilidad realizada rolling de 30 barras, que son unos 5 días
                      de trading (una semana). Da una visión algo más estable.
- range_proxy_1h    : proxy de rango por barra: (High - Low) / Close. Mide cuánto se
                      ha movido el precio dentro de cada hora, algo que en datos diarios
                      no se puede ver.
- abs_return        : valor absoluto del rendimiento horario, proxy directo de volatilidad.
- sq_return         : rendimiento al cuadrado, la base clásica del modelo ARCH.
- log_volume_change : cambio logarítmico en el volumen de una hora a la siguiente. Sirve
                      como proxy de la actividad del mercado en tiempo real.
- return_fwd_1h     : el retorno de la siguiente barra, que usamos como target predictivo
                      a 1 hora vista.
- return_fwd_3h     : retorno 3 barras adelante, target a 3 horas vista.
- vol_fwd_6h        : volatilidad realizada de las próximas 6 barras, otro target
                      predictivo pero orientado a volatilidad en vez de retorno.
- date              : la fecha simple (YYYY-MM-DD) para poder cruzar con datos diarios
                      de noticias y sentimiento.
- hora              : la hora del día en ET (9-15), para capturar patrones intradía
                      típicos como los picos de volatilidad en apertura y cierre.
- dia_semana        : día de la semana (0=lunes), por si hay efecto de fin de semana.
- es_sesion_regular : True cuando la barra cae dentro del horario de mercado normal
                      (9:30 a 16:00 hora del Este).

Sobre el factor de anualización
-------------------------------
Para datos horarios usamos sqrt(252 * 6.5) que da aproximadamente 40.47. El 252
son los días hábiles del año y 6.5 son las horas que dura la sesión de mercado
(de 09:30 a 16:00 ET). Así las volatilidades anualizadas son directamente
comparables con las de la base diaria.

Uso
---
    source venv/bin/activate
    python scripts/download_market_intraday.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — parámetros de la descarga y cálculo de variables
# ─────────────────────────────────────────────────────────────────────────────

TICKERS      = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
INTERVALO    = "1h"

# yfinance limita los datos horarios a unos ~730 días (verificado empíricamente).
# Pedimos period='2y' para descargar el máximo posible sin preocuparnos por fechas exactas.
PERIOD       = "2y"

# Factor para pasar volatilidad horaria a anualizada:
# 252 días hábiles al año por 6.5 horas de mercado cada día = 1638 barras/año
FACTOR_ANUAL = np.sqrt(252 * 6.5)  # sale aproximadamente 40.47

# Ventanas rolling pensadas para escala horaria:
# - 6 barras son más o menos 1 día de trading (la sesión dura 6.5h, redondeamos a 6)
# - 30 barras son unos 5 días de trading, o sea una semana bursátil
VENTANA_CORTA  = 6    # aproximadamente 1 día de mercado
VENTANA_LARGA  = 30   # aproximadamente 1 semana de mercado

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_SALIDA    = RAIZ_PROYECTO / "data" / "raw" / "market_intraday"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES — cada paso del pipeline de descarga y cálculo
# ─────────────────────────────────────────────────────────────────────────────

def descargar_ohlcv_horario(ticker: str) -> pd.DataFrame:
    """
    Se conecta a Yahoo Finance y descarga los precios OHLCV con granularidad
    de 1 hora para una empresa concreta. Yahoo devuelve como máximo ~2 años de
    datos horarios, así que la ventana temporal queda determinada automáticamente.

    Parámetros
    ----------
    ticker : str  — el símbolo bursátil, por ejemplo 'AAPL' para Apple.

    Devuelve
    --------
    Un DataFrame de pandas con las columnas open, high, low, close y volume.
    El índice es un datetime con zona horaria America/New_York (Eastern Time),
    que es la zona del mercado de Nueva York.
    """
    datos = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVALO,
        auto_adjust=True,   # ajuste automático de splits y dividendos
        progress=False,
    )

    if datos.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    # A veces yfinance devuelve columnas con MultiIndex, así que lo aplanamos por si acaso
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    datos.columns = [c.lower() for c in datos.columns]

    # Nos aseguramos de que todo esté en Eastern Time (zona del mercado de NY)
    if datos.index.tz is not None:
        datos.index = datos.index.tz_convert("America/New_York")
    else:
        datos.index = datos.index.tz_localize("UTC").tz_convert("America/New_York")

    datos.index.name = "datetime"
    return datos[["open", "high", "low", "close", "volume"]].sort_index()


def calcular_variables(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    A partir de los precios horarios en crudo, calcula todas las variables derivadas
    que necesitamos para los modelos. Es parecido a lo que hacemos con datos diarios,
    pero adaptado a escala horaria: las ventanas rolling son más cortas, el factor de
    anualización es diferente, y añadimos variables temporales (hora, día de la semana)
    que tienen sentido en datos intradía.

    Parámetros
    ----------
    df     : DataFrame con las columnas OHLCV originales (índice = datetime horario con tz).
    ticker : el símbolo bursátil, que se mete como columna identificadora.

    Devuelve
    --------
    El mismo DataFrame pero con todas las variables calculadas añadidas.
    """
    df = df.copy()

    # Rendimiento logarítmico por cada barra de 1 hora
    df["log_return_1h"] = np.log(df["close"] / df["close"].shift(1))

    # Volatilidad realizada rolling, anualizada usando el factor horario
    df[f"vol_realized_{VENTANA_CORTA}h"] = (
        df["log_return_1h"].rolling(VENTANA_CORTA).std() * FACTOR_ANUAL
    )
    df[f"vol_realized_{VENTANA_LARGA}h"] = (
        df["log_return_1h"].rolling(VENTANA_LARGA).std() * FACTOR_ANUAL
    )

    # Cuánto se ha movido el precio dentro de cada hora: (máximo - mínimo) / cierre
    df["range_proxy_1h"] = (df["high"] - df["low"]) / df["close"]

    # Proxies clásicos de volatilidad: el absoluto y el cuadrado del rendimiento
    df["abs_return"] = df["log_return_1h"].abs()
    df["sq_return"]  = df["log_return_1h"] ** 2

    # Cambio logarítmico en volumen de una hora a la siguiente
    df["log_volume_change"] = np.log(df["volume"] / df["volume"].shift(1))

    # --- Variables forward (lo que queremos predecir) ---
    # shift(-N) "mira hacia el futuro", poniendo en cada fila el valor de N barras después
    df["return_fwd_1h"] = df["log_return_1h"].shift(-1)
    df["return_fwd_3h"] = np.log(df["close"].shift(-3) / df["close"])
    df["vol_fwd_6h"]    = (
        df["log_return_1h"].shift(-VENTANA_CORTA)
        .rolling(VENTANA_CORTA).std() * FACTOR_ANUAL
    )
    # Para calcular correctamente la volatilidad forward: en la fila t queremos la
    # volatilidad de las barras t+1 hasta t+6, así que recorremos con un bucle
    returns = df["log_return_1h"].values
    vol_fwd = np.full(len(returns), np.nan)
    for i in range(len(returns) - VENTANA_CORTA):
        ventana = returns[i + 1 : i + 1 + VENTANA_CORTA]
        if not np.any(np.isnan(ventana)):
            vol_fwd[i] = np.std(ventana, ddof=1) * FACTOR_ANUAL
    df["vol_fwd_6h"] = vol_fwd

    # Fecha simple para poder cruzar con datos diarios de noticias y sentimiento
    df["date"] = df.index.date

    # Variables temporales para capturar patrones intradía (el índice ya está en ET)
    df["hora"]       = df.index.hour
    df["dia_semana"] = df.index.dayofweek  # 0=lunes, 4=viernes

    # Marcamos si cada barra cae dentro del horario regular de mercado (09:30-16:00 ET).
    # Las barras de yfinance marcan el INICIO del intervalo, así que la barra de las 9:30
    # tiene hora=9, y la última de la sesión regular (15:30) tiene hora=15.
    hora_dt = df.index.hour + df.index.minute / 60
    df["es_sesion_regular"] = (hora_dt >= 9.5) & (hora_dt < 16.0)

    # Metemos el ticker como primera columna para identificar la empresa
    df.insert(0, "ticker", ticker)

    return df


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un DataFrame en formato CSV manteniendo el datetime como índice."""
    df.to_csv(ruta, index=True, date_format="%Y-%m-%d %H:%M:%S%z")
    log.info(f"Guardado → {ruta.relative_to(RAIZ_PROYECTO)}")


def imprimir_resumen(df_all: pd.DataFrame) -> None:
    """Imprime un resumen completo de lo descargado: cobertura temporal, estadísticas
    de rendimientos horarios, volatilidad media y valores nulos desglosados por empresa."""
    sep = "─" * 80

    print(f"\n{sep}")
    print("  RESUMEN DE DATOS INTRADÍA DESCARGADOS — Magnificent 7 (1h)")
    print(sep)

    grupos = df_all.groupby("ticker")

    # Cuántas barras tiene cada empresa y de qué fecha a qué fecha
    print("\n  COBERTURA TEMPORAL\n")
    print(f"  {'Ticker':<8} {'Barras':>7}  {'Inicio':>22}  {'Fin':>22}")
    print(f"  {'------':<8} {'------':>7}  {'--------------------':>22}  {'--------------------':>22}")
    for ticker, g in grupos:
        print(
            f"  {ticker:<8} {len(g):>7}  "
            f"{str(g.index.min()):>22}  "
            f"{str(g.index.max()):>22}"
        )

    # Estadísticas de los rendimientos logarítmicos horarios
    print("\n  RENDIMIENTOS LOGARÍTMICOS HORARIOS\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        r = g["log_return_1h"].dropna()
        print(f"  {ticker:<8} {r.mean():>10.7f}  {r.std():>10.7f}  {r.min():>10.6f}  {r.max():>10.6f}")

    # Volatilidad realizada con ventana de 30 barras (aprox. 1 semana)
    print(f"\n  VOLATILIDAD REALIZADA MEDIA (ventana {VENTANA_LARGA}h, anualizada)\n")
    print(f"  {'Ticker':<8} {'Media':>10}  {'Std':>10}  {'Max':>10}")
    print(f"  {'------':<8} {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for ticker, g in grupos:
        v = g[f"vol_realized_{VENTANA_LARGA}h"].dropna()
        print(f"  {ticker:<8} {v.mean():>10.4f}  {v.std():>10.4f}  {v.max():>10.4f}")

    # Revisión de valores nulos (los primeros de cada ventana rolling son NaN, eso es esperado)
    print("\n  VALORES NULOS POR COLUMNA\n")
    nulos = df_all.isnull().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print("  Ninguno inesperado.")
    else:
        for col, n in nulos.items():
            pct = 100 * n / len(df_all)
            print(f"  {col:<30} {n:>6} nulos  ({pct:.1f}%)")

    print(f"\n{sep}")
    print(
        f"  Total barras: {len(df_all):,}  |  "
        f"Empresas: {df_all['ticker'].nunique()}  |  "
        f"Intervalo: {INTERVALO}"
    )
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — descarga, calcula variables y guarda todo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Inicio descarga de datos intradía (1h) ===")
    log.info(f"Periodo: {PERIOD} máximo disponible | Intervalo: {INTERVALO} | Empresas: {', '.join(TICKERS)}")

    frames = []

    for ticker in tqdm(TICKERS, desc="Descargando", unit="ticker"):
        try:
            df_raw = descargar_ohlcv_horario(ticker)
            df     = calcular_variables(df_raw, ticker)

            nombre = f"{ticker}_market_intraday_2024_2026.csv"
            guardar_csv(df, DIR_SALIDA / nombre)
            frames.append(df)

            log.info(
                f"{ticker}: {len(df):,} barras | "
                f"{df.index.min().date()} → {df.index.max().date()}"
            )
        except Exception as e:
            log.error(f"Error con {ticker}: {e}")
            sys.exit(1)

    # Juntamos todas las empresas en un solo CSV combinado
    df_all = pd.concat(frames).sort_values(["ticker", "datetime"])
    guardar_csv(df_all, DIR_SALIDA / "ALL_market_intraday_2024_2026.csv")
    log.info(f"CSV combinado: {len(df_all):,} barras en total")

    imprimir_resumen(df_all)
    log.info("=== Descarga completada ===")


if __name__ == "__main__":
    main()
