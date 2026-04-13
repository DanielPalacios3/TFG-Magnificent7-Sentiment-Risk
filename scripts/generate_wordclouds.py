"""
generate_wordclouds.py
======================
Aquí se crean las nubes de palabras (wordclouds) para cada una de las 7 empresas
Magnificent 7. La idea es visualizar qué términos y temas aparecen más en las
discusiones y noticias sobre cada empresa. Se generan dos tipos de wordcloud:

  - Reddit: usa el texto completo de los posts (campo texto_completo de
    reddit_clean_posts.csv). Aquí es donde se ve de qué habla la gente en
    r/wallstreetbets, r/stocks, etc. sobre cada ticker.

  - Noticias GDELT: usa el campo domain de news_clean_articles.csv. Aquí hay que
    explicar algo: de GDELT solo tenemos metadatos (fecha, url, domain, tono,
    organizaciones...), no descargamos el texto completo de las noticias porque era
    inviable. Así que en vez de una nube de "temas", lo que se muestra es una nube
    de "fuentes" — qué medios de comunicación hablan más de cada empresa. Es una
    forma útil de ver, por ejemplo, si Tesla sale más en medios tech o generalistas.

Las figuras se guardan en:
  - docs/figuras/wordcloud_reddit_<TICKER>.png
  - docs/figuras/wordcloud_noticias_<TICKER>.png
"""

import re
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
FIGURAS = Path("docs/figuras")
FIGURAS.mkdir(parents=True, exist_ok=True)

# Stopwords: además de las típicas en inglés, añadimos términos financieros genéricos
# que aparecen en todos los posts y no aportan nada interesante a la nube (como "stock",
# "buy", "sell"...). También metemos los propios tickers y nombres de empresas porque
# si no, dominarían toda la nube y no veríamos nada más.
STOPWORDS_EXTRA = {
    "stock", "stocks", "share", "shares", "market", "trading", "trade",
    "price", "buy", "sell", "company", "companies", "year", "time",
    "week", "month", "day", "today", "like", "just", "think", "get",
    "one", "would", "could", "will", "going", "know", "good", "make",
    "still", "even", "much", "way", "well", "also", "back", "new",
    "https", "www", "com", "http", "reddit", "deleted", "removed",
    # tickers y nombres de empresas (si no los quitamos, acaparan toda la nube)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "aapl", "msft", "googl", "amzn", "nvda", "meta", "tsla",
    "apple", "microsoft", "google", "amazon", "nvidia", "tesla",
    # cashtags con el símbolo de dólar (muy comunes en Reddit financiero)
    "$AAPL", "$MSFT", "$GOOGL", "$AMZN", "$NVDA", "$META", "$TSLA",
}
STOP = STOPWORDS.union(STOPWORDS_EXTRA)


def limpiar_texto(texto: str) -> str:
    """Limpia un texto para que quede listo para la nube de palabras: quita URLs,
    cashtags ($AAPL y similares), menciones (@usuario), y cualquier carácter que no
    sea una letra. Al final todo queda en minúsculas y sin espacios de más."""
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"\$\w+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.lower().strip()


def generar_wordcloud(texto: str, titulo: str, ruta_salida: Path, color: str = "Blues"):
    """Crea una nube de palabras a partir de un texto ya limpio, le pone un título bonito
    y la guarda como imagen PNG. Se puede cambiar la paleta de colores con el parámetro color."""
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=color,
        stopwords=STOP,
        max_words=150,
        collocations=False,
        min_word_length=3,
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_salida.name}")


# ---------------------------------------------------------------------------
# TAREA 1 — Word clouds de Reddit
# Cogemos todos los posts de cada empresa y creamos una nube con las palabras
# más frecuentes. Esto da una idea rápida de qué temas dominan la conversación.
# ---------------------------------------------------------------------------
print("\n=== Word clouds Reddit ===")
reddit = pd.read_csv("data/clean/reddit_clean_posts.csv", usecols=["ticker", "texto_completo"])
reddit["texto_completo"] = reddit["texto_completo"].fillna("").astype(str)

for ticker in TICKERS:
    df = reddit[reddit["ticker"] == ticker]
    if df.empty:
        print(f"  {ticker}: sin posts — saltando")
        continue
    corpus = " ".join(df["texto_completo"].tolist())
    corpus = limpiar_texto(corpus)
    if len(corpus.strip()) < 10:
        print(f"  {ticker}: texto vacío tras limpieza — saltando")
        continue
    titulo = f"Reddit — Términos más frecuentes: {ticker}"
    ruta = FIGURAS / f"wordcloud_reddit_{ticker}.png"
    generar_wordcloud(corpus, titulo, ruta, color="Blues")

print("Reddit: completado")

# ---------------------------------------------------------------------------
# TAREA 2 — Word clouds de GDELT (por dominio/fuente de noticias)
# Como no tenemos el texto completo de las noticias (solo metadatos), usamos
# los dominios de los medios para ver qué fuentes cubren más cada empresa.
# Por ejemplo, reuters aparecerá grande si publica mucho sobre esa empresa.
# ---------------------------------------------------------------------------
print("\n=== Word clouds GDELT (dominios de fuentes) ===")
# Solo leemos las columnas que necesitamos porque el CSV de noticias tiene 6,5M filas
# y cargar todo en memoria sería un desperdicio de RAM
noticias = pd.read_csv(
    "data/clean/news_clean_articles.csv",
    usecols=["fecha", "domain", "query_ticker"],
    low_memory=True,
)
noticias["domain"] = noticias["domain"].fillna("").astype(str)

for ticker in TICKERS:
    df = noticias[noticias["query_ticker"] == ticker]
    if df.empty:
        print(f"  {ticker}: sin artículos — saltando")
        continue
    # La idea: juntamos todos los dominios en un solo string largo tipo
    # "reuters reuters bbc bloomberg bloomberg bloomberg..." y dejamos que la
    # wordcloud cuente frecuencias. Quitamos los TLDs (.com, .co.uk, etc.)
    # para que queden solo los nombres limpios de los medios
    corpus = " ".join(df["domain"].tolist())
    corpus = re.sub(r"\.(com|net|org|co|uk|io|eu|de|fr|es|ru|cn|jp|au|ca|in)\b", " ", corpus)
    corpus = re.sub(r"[^a-zA-Z\s]", " ", corpus)
    corpus = re.sub(r"\s+", " ", corpus).lower().strip()
    if len(corpus.strip()) < 10:
        print(f"  {ticker}: texto vacío tras limpieza — saltando")
        continue
    titulo = f"GDELT Noticias — Fuentes más citadas: {ticker}"
    ruta = FIGURAS / f"wordcloud_noticias_{ticker}.png"
    generar_wordcloud(corpus, titulo, ruta, color="Oranges")

print("GDELT: completado")
print("\nTodas las word clouds generadas en docs/figuras/")
