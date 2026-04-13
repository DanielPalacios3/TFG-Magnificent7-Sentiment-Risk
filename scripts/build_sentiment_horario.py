"""
Script: build_sentiment_horario.py
Proyecto: TFG — Predicción del riesgo de mercado en las Magnificent 7
Autor: Daniel Palacios García — UFV Madrid

Este script resuelve un problema del dataset intradía: en build_dataset_intraday.py
todas las barras de un mismo día comparten el mismo sentimiento diario, lo cual no
es ideal. Lo que hacemos aquí es calcular un sentimiento ACUMULADO para cada barra
horaria, de forma que cada hora del día tenga un valor diferente.

La idea es simple pero potente: para la barra de las 9:30, acumulamos todos los posts
de Reddit publicados desde el cierre del mercado del día anterior (16:00 ET) hasta las
9:30. Así capturamos el sentimiento overnight. Para la barra de las 10:30, sumamos
también los posts de esa primera hora del día. Y así sucesivamente. Al llegar a la
barra de las 15:30, tenemos acumulados casi todos los posts del día.

El sentimiento se calcula usando las probabilidades de FinBERT (que ya fueron
generadas previamente), no el tono bruto de GDELT.

Archivo que genera
------------------
- data/clean/reddit_sentimiento_horario.csv — una fila por barra con el sentimiento
  acumulado desde el cierre anterior

Las variables que genera son:
- rddt_acum_n_posts: cuántos posts se han acumulado desde el cierre de ayer
- rddt_acum_sent_score: la media de (probabilidad_positiva - probabilidad_negativa)
  de FinBERT, que va de -1 (muy negativo) a +1 (muy positivo)
- rddt_acum_pct_neg: qué porcentaje de los posts acumulados son negativos
- rddt_acum_pct_pos: qué porcentaje de los posts acumulados son positivos
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']

print("=" * 70)
print("CONSTRUYENDO SENTIMIENTO HORARIO ACUMULADO")
print("=" * 70)

# 1. Cargamos los posts en crudo (necesitamos el timestamp original created_utc)
# y los scores de FinBERT que ya calculamos previamente. Los unimos por id.
raw = pd.read_csv(RAIZ / 'data' / 'raw' / 'reddit' / 'ALL_reddit_2019_2026.csv', low_memory=False)
finbert = pd.read_csv(RAIZ / 'data' / 'clean' / 'reddit_finbert_posts.csv')

raw['datetime'] = pd.to_datetime(raw['created_utc'], unit='s', utc=True)
raw['datetime_et'] = raw['datetime'].dt.tz_convert('US/Eastern')

# Juntamos cada post con su score de FinBERT usando el id como clave
posts = raw.merge(
    finbert[['id', 'finbert_positive', 'finbert_negative', 'finbert_label']],
    on='id', how='inner'
)

# Nos quedamos solo con los posts del periodo intradía (2024 en adelante)
posts = posts[posts['datetime'] >= '2024-04-01'].copy()
print(f"Posts con FinBERT y hora (2024+): {len(posts):,}")

# 2. Cargamos las barras intradía del dataset integrado para saber en qué
# momentos exactos necesitamos calcular el sentimiento acumulado
df_intra = pd.read_csv(RAIZ / 'data' / 'clean' / 'dataset_intraday_integrado.csv')
df_intra['datetime'] = pd.to_datetime(df_intra['datetime'], utc=True).dt.tz_localize(None)
print(f"Barras intradía: {len(df_intra):,}")

# 3. El bucle principal: para cada barra horaria de cada ticker, buscamos
# todos los posts publicados entre el cierre del día anterior (16:00 ET)
# y el inicio de esa barra. Así cada barra tiene su propio sentimiento
# acumulado, que va creciendo a lo largo del día.
resultados = []

for ticker in TICKERS:
    posts_t = posts[posts['ticker'] == ticker].copy()
    barras_t = df_intra[df_intra['ticker'] == ticker].sort_values('datetime')
    
    for _, barra in barras_t.iterrows():
        barra_dt = barra['datetime']
        barra_hora = barra['hora']
        barra_date = barra_dt.date()
        
        # Calculamos la ventana temporal: desde las 16:00 ET del día hábil
        # anterior hasta la hora de inicio de esta barra
        dia_anterior = barra_date - pd.Timedelta(days=1)
        while dia_anterior.weekday() >= 5:  # Saltamos fines de semana
            dia_anterior -= pd.Timedelta(days=1)
        
        cierre_anterior = pd.Timestamp(dia_anterior, tz='US/Eastern').replace(hour=16, minute=0)
        barra_actual = pd.Timestamp(barra_date, tz='US/Eastern').replace(hour=barra_hora, minute=30)
        
        mask = (posts_t['datetime_et'] >= cierre_anterior) & (posts_t['datetime_et'] < barra_actual)
        ventana = posts_t[mask]
        
        n_posts = len(ventana)
        if n_posts > 0:
            sent_score = (ventana['finbert_positive'] - ventana['finbert_negative']).mean()
            pct_neg = (ventana['finbert_label'] == 'negative').mean()
            pct_pos = (ventana['finbert_label'] == 'positive').mean()
        else:
            sent_score = np.nan
            pct_neg = np.nan
            pct_pos = np.nan
        
        resultados.append({
            'datetime': barra_dt,
            'ticker': ticker,
            'rddt_acum_n_posts': n_posts,
            'rddt_acum_sent_score': sent_score,
            'rddt_acum_pct_neg': pct_neg,
            'rddt_acum_pct_pos': pct_pos,
        })

df_sent = pd.DataFrame(resultados)

# Guardamos el resultado final
ruta = RAIZ / 'data' / 'clean' / 'reddit_sentimiento_horario.csv'
df_sent.to_csv(ruta, index=False)

print(f"\nGuardado: {ruta}")
print(f"Filas: {len(df_sent):,}")
print(f"Nulos: {df_sent['rddt_acum_sent_score'].isna().sum()} ({df_sent['rddt_acum_sent_score'].isna().mean()*100:.0f}%)")
print(f"\nMedia posts acumulados por barra:")
for ticker in TICKERS:
    dt = df_sent[df_sent['ticker'] == ticker]
    print(f"  {ticker}: {dt['rddt_acum_n_posts'].mean():.1f} posts/barra")
