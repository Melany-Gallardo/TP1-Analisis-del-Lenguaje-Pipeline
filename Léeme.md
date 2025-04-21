# 📊 Análisis de Lenguajes de Programación con TF-IDF

## 🎯 Objetivo
Aplicar procesamiento de lenguaje natural (NLP) usando TF-IDF para analizar el contenido del archivo `CorpusLenguajes.txt`, siguiendo una **canalización (pipeline)** completa.

---

## ✅ Consigna

Realizar una canalización para analizar el corpus:
- Aplicar `stopwords`
- Lemmatizar
- Calcular TF-IDF
- Mostrar:
  - El corpus preparado
  - La matriz TF-IDF generada
  - El vocabulario generado

Luego, se deben analizar los resultados:
- Obtener las 6 palabras más usadas en todo el corpus
- Detectar:
  - La palabra menos utilizada
  - Las palabras más repetidas en la misma oración
- Imprimir un gráfico de distribución de frecuencia

---

## ⚙️ Tecnologías utilizadas

- Python 3.11+
- `nltk` (Para procesamiento del texto)
- `scikit-learn` (`sklearn` para TF-IDF)
- `seaborn` (para gráficos/visualización)
- `re` (expresiones regulares)
- `string` y `collections` (para manejo de texto)

---


## ▶️ Instrucciones para ejecutar

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt

---

## 🧪 Instalación y ejecución

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/tp1-tfidf.git
cd tp1-tfidf
