# -*- coding: utf-8 -*-
"""
Módulo de preprocesamiento para datos financieros y texto
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class DataPreprocessor:
    """Clase para preprocesar datos financieros y de texto"""
    
    def __init__(self, language='spanish'):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=1000)
        
    def load_data(self, filepath):
        """Cargar dataset desde CSV"""
        df = pd.read_csv(filepath, sep=';')
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        return df
    
    def create_financial_features(self, df):
        """Crear variables financieras derivadas"""
        df = df.copy()
        df['daily_return'] = (df['close'] - df['open']) / df['open']
        df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
        df['volume_scaled'] = df['volume'] / df['volume'].max()
        df['price_range'] = df['high'] - df['low']
        return df
    
    def clean_text(self, text):
        """Limpiar y normalizar texto"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return ' '.join(words)
    
    def prepare_text_features(self, df, column='Headline_esp'):
        """Preparar características de texto con TF-IDF"""
        df = df.copy()
        df['text_clean'] = df[column].astype(str).apply(self.clean_text)
        X_text = self.tfidf.fit_transform(df['text_clean'])
        return X_text, df
    
    def prepare_numeric_features(self, df, features):
        """Preparar características numéricas con escalado"""
        X_num = df[features].values
        X_num_scaled = self.scaler.fit_transform(X_num)
        return X_num_scaled
    
    def get_feature_names(self):
        """Obtener nombres de las características TF-IDF"""
        return self.tfidf.get_feature_names_out()