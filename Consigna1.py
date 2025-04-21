# ---------------------------------------
# Consigna 1:Canalización del texto para realizar un análisis del texto.
# ---------------------------------------

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns #usé seaborn para mejorar la visualización de los gráficos y porque es más estético
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Eliminar stopwords
def quitarStopwords_eng(tokens):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in tokens if word.lower() not in stop_words and word.isalnum()]

# Lematizar
lemmatizer = WordNetLemmatizer()
def lematizar(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

corpus = [
    lematizar(quitarStopwords_eng(nltk.word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
    lematizar(quitarStopwords_eng(nltk.word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]

# Muestra el corpus preparado
print("Corpus procesado:")
for i, doc in enumerate(corpus):
    print(f"Documento {i+1}:", doc)

# Convierto el corpus a string para el TF-IDF
corpus_str = [" ".join(doc) for doc in corpus]

# Aplico TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_str)

# Muestra la matriz TF-IDF generada
print("\n Matriz TF-IDF generada:")
print(tfidf_matrix.toarray())

# Muestra el vocabulario generado
print("\n Vocabulario generado:")
print(vectorizer.get_feature_names_out())

# ---------------------------------------
# Análisis del corpus y gráfico
# ---------------------------------------

# 1. Jerarquía de las 6 palabras más usadas
print("\n Top 6 palabras más frecuentes (suma TF-IDF):")
suma_tfidf = tfidf_matrix.sum(axis=0).A1
palabras = vectorizer.get_feature_names_out()
top6 = sorted(zip(palabras, suma_tfidf), key=lambda x: x[1], reverse=True)[:6]
for palabra, valor in top6:
    print(f"{palabra}: {valor:.4f}")

# 2. Palabra menos utilizada
print("\n Palabra menos utilizada (menor TF-IDF total):")
menos_usada = sorted(zip(palabras, suma_tfidf), key=lambda x: x[1])[0]
print(f"{menos_usada[0]}: {menos_usada[1]:.4f}")

# 3. Palabras más repetidas en una misma oración
print("\n Palabras más repetidas en una misma oración (una por línea):")
for idx, fila in enumerate(tfidf_matrix.toarray()):
    i_max = fila.argmax()
    print(f"Documento {idx+1}: {palabras[i_max]} (TF-IDF: {fila[i_max]:.4f})")

# 4. Gráfico de distribución de frecuencia de palabras
print("\n Mostrando gráfico de distribución de frecuencia de palabras...")

plt.figure(figsize=(12, 6))
sns.barplot(x=[x[0] for x in top6], y=[x[1] for x in top6], palette="magma")
plt.title("Jerarquía de palabras más usadas (TF-IDF)")
plt.ylabel("Peso TF-IDF")
plt.xlabel("Palabra")
plt.tight_layout()
plt.show()
