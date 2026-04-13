# Predicción del riesgo de mercado en las Magnificent 7 con sentimiento de noticias y redes sociales

**Trabajo Fin de Grado** — Universidad Francisco de Vitoria  
**Autor:** Daniel Palacios García  
**Grado:** Business Analytics 
**Curso:** 2025-2026

---

## Sobre el proyecto

Este proyecto investiga si el sentimiento extraído de noticias financieras y redes sociales aporta información útil para predecir la volatilidad de las acciones de las Magnificent 7 (Apple, Microsoft, Alphabet, Amazon, NVIDIA, Meta y Tesla).

Se construye un dataset que combina tres fuentes:
- **Precios de mercado** de Yahoo Finance (2019-2026): 12.754 observaciones diarias + 24.430 barras horarias
- **Noticias financieras** de GDELT vía Google BigQuery: 6,76 millones de artículos con tono de sentimiento
- **Publicaciones de Reddit** vía Arctic Shift: ~100.000 posts de r/wallstreetbets, r/stocks, r/investing, r/StockMarket y r/options, procesados con FinBERT

Se comparan tres modelos a dos escalas temporales (diaria e intradía):

| Modelo | Descripción | Mejor en... |
|--------|-------------|-------------|
| GARCH(1,1) | Modelo econométrico clásico de volatilidad | Escala diaria (RMSE 0,071) |
| GARCH-X | GARCH + sentimiento como variable exógena | Significativo en intradía (13/14 coeficientes) |
| XGBoost | Machine learning con 26 variables | Escala intradía (RMSE 0,167) |

### Hallazgo principal

El sentimiento tiene poder predictivo real a escala intradía (horas), pero a escala diaria ya ha sido absorbido por el mercado al cierre. En intradía, el sentimiento aporta el 41,8% de la importancia de variables de XGBoost y mejora significativamente a GARCH en 5 de 7 empresas (test de Diebold-Mariano).

---

## Estructura del repositorio

```
├── data/clean/                     ← Datasets procesados (los que usan los notebooks)
│   ├── dataset_integrado.csv       ← Dataset diario (7 empresas × 1.822 días, 39 variables)
│   ├── dataset_intraday_integrado.csv  ← Dataset intradía (24.430 barras, 42 variables)
│   ├── news_clean_daily.csv        ← Tono GDELT agregado por día y empresa
│   ├── reddit_clean_daily.csv      ← Actividad Reddit agregada por día y empresa
│   ├── reddit_clean_posts.csv      ← Posts individuales de Reddit (limpios)
│   └── reddit_sentimiento_horario.csv  ← Sentimiento Reddit acumulado por hora
│
├── scripts/                        ← Scripts de extracción, limpieza e integración
│   ├── download_market.py          ← Descarga precios diarios de Yahoo Finance
│   ├── download_market_intraday.py ← Descarga barras horarias
│   ├── download_news_bigquery.py   ← Extrae noticias de GDELT vía BigQuery
│   ├── download_reddit.py          ← Descarga posts de Reddit vía Arctic Shift
│   ├── clean_market.py             ← Limpieza datos de mercado
│   ├── clean_news.py               ← Limpieza datos GDELT
│   ├── clean_reddit.py             ← Limpieza posts de Reddit
│   ├── build_dataset.py            ← Integración del dataset diario
│   ├── build_dataset_intraday.py   ← Integración del dataset intradía
│   ├── apply_finbert_reddit.py     ← Clasificación de sentimiento con FinBERT
│   ├── build_sentiment_horario.py  ← Sentimiento acumulado por hora
│   ├── generate_wordclouds.py      ← Nubes de palabras por empresa
│   └── generar_diagnosticos.py     ← Figuras de diagnóstico estadístico
│
├── notebooks/                      ← Análisis y modelos
│   ├── eda_mercado.ipynb           ← EDA de datos de mercado
│   ├── eda_texto.ipynb             ← EDA de sentimiento (GDELT + Reddit)
│   ├── eda_intraday.ipynb          ← EDA de patrones intradía
│   ├── eda_intraday_texto.ipynb    ← EDA de sentimiento intradía
│   ├── modelos.ipynb               ← Modelos diarios: GARCH, GARCH-X, XGBoost
│   └── modelos_intraday.ipynb      ← Modelos intradía + Diebold-Mariano
│
└── docs/figuras/                   ← Todas las figuras generadas
```

---

## Cómo ejecutar

### Requisitos

```bash
pip install pandas numpy scipy matplotlib seaborn yfinance arch xgboost transformers torch statsmodels wordcloud scikit-learn
```

Para la descarga de noticias de GDELT también se necesita:
```bash
pip install google-cloud-bigquery
gcloud auth application-default login
```

### Sobre los datos

El repositorio incluye los datasets procesados en `data/clean/` que son los que usan los notebooks directamente. Los datos en crudo (raw) no se incluyen porque pesan casi 8 GB (solo las noticias de GDELT ocupan 6 GB) y superan los límites de almacenamiento de GitHub. Si quieres reproducir el proyecto desde la extracción, los scripts de descarga (`download_*.py`) permiten obtener todos los datos originales desde las APIs públicas de Yahoo Finance, GDELT BigQuery y Arctic Shift.

### Reproducir el análisis

Los notebooks cargan directamente los datos de `data/clean/`. Para ejecutarlos:

```bash
jupyter notebook notebooks/eda_mercado.ipynb
jupyter notebook notebooks/modelos.ipynb
```

Si quieres reproducir todo desde cero (extracción → limpieza → integración → modelos):

```bash
# 1. Extracción de datos
python scripts/download_market.py
python scripts/download_news_bigquery.py
python scripts/download_reddit.py

# 2. Limpieza
python scripts/clean_market.py
python scripts/clean_news.py
python scripts/clean_reddit.py

# 3. FinBERT
python scripts/apply_finbert_reddit.py

# 4. Integración
python scripts/build_dataset.py
python scripts/build_dataset_intraday.py
python scripts/build_sentiment_horario.py

# 5. Ejecutar notebooks de análisis y modelos
```

---

## Fuentes de datos

| Fuente | Acceso | Coste |
|--------|--------|-------|
| [Yahoo Finance](https://finance.yahoo.com/) | API pública (yfinance) | Gratuito |
| [GDELT](https://www.gdeltproject.org/) | Google BigQuery | Gratuito (1 TB/mes) |
| [Reddit (Arctic Shift)](https://arctic-shift.photon-reddit.com/) | REST API | Gratuito |
| [FinBERT](https://huggingface.co/ProsusAI/finbert) | Hugging Face | Gratuito |

---

## Resultados principales

- **Escala diaria:** GARCH(1,1) gana con RMSE 0,071. El sentimiento solo aporta un +0,4% (ablación).
- **Escala intradía:** XGBoost gana con RMSE 0,167 (mejora 7,6% sobre GARCH). El sentimiento aporta +1,9% y representa el 41,8% de la importancia.
- **Empresas más sensibles al sentimiento:** TSLA (DM = +5,54) y NVDA (+4,91).
- **Fuente más importante:** Reddit (24,2%) > FinBERT (9,1%) > GDELT (8,5%).

---

## Referencias principales

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
- Baker, M. y Wurgler, J. (2006). Investor Sentiment and the Cross-Section of Stock Returns.
- Tetlock, P. C. (2007). Giving Content to Investor Sentiment.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
- Long, C. et al. (2022). The Role of Reddit in the GameStop Short Squeeze.
- Chen, T. y Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
