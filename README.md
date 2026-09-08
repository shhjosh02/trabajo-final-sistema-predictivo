# 🚀 Sistema Híbrido de Alertas Tempranas para Predicción Financiera

## 📊 Descripción

Este proyecto implementa un sistema de inteligencia artificial que combina:

- **Análisis de datos financieros** (Pandas, NumPy)
- **Procesamiento de Lenguaje Natural** (NLTK, SciPy)
- **Machine Learning** (Scikit-learn, PyTorch)
- **Deep Learning** (TensorFlow, Keras)

El objetivo es clasificar si una noticia financiera provocará un movimiento al alza (1) o a la baja (0) en el precio de una acción.

## 📁 Estructura del Proyecto
trabajo-final-sistema-predictivo/
├── data/ # Dataset con acciones y noticias
├── notebooks/ # 3 notebooks con todo el código
├── src/ # Funciones reutilizables
├── outputs/ # Gráficos generados
├── models/ # Modelo LSTM guardado
├── requirements.txt
├── README.md
└── .gitignore

text

## 🛠️ Tecnologías Utilizadas

| Herramienta | Uso |
|-------------|-----|
| Pandas / NumPy | Manipulación de datos |
| Matplotlib / Seaborn | Visualización |
| NLTK / SciPy | Procesamiento de texto |
| Scikit-learn | Modelos de ML |
| PyTorch | Red neuronal numérica |
| TensorFlow / Keras | Modelo LSTM |

## 📋 Fases del Proyecto

| Fase | Descripción | Herramientas |
|------|-------------|--------------|
| 1 | Carga y limpieza de datos | Pandas, NumPy |
| 2 | Análisis exploratorio (EDA) | Matplotlib, Seaborn |
| 3 | Procesamiento de lenguaje (NLP) | NLTK, SciPy |
| 4 | Modelos de Machine Learning | Scikit-learn, PyTorch |
| 5 | Deep Learning (LSTM) | TensorFlow, Keras |
| 6 | Evaluación y visualización | Scikit-learn, Matplotlib |

## 🚀 Cómo Ejecutar

### En Google Colab (recomendado):
1. Abrir https://colab.research.google.com/
2. Subir los notebooks de la carpeta `notebooks/`
3. Ejecutar en orden: 1, 2, 3

### En local:
```bash
pip install -r requirements.txt
jupyter notebook