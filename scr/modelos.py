# -*- coding: utf-8 -*-
"""
Módulo para modelos de Machine Learning y Deep Learning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLModels:
    """Clase para modelos de Machine Learning con Scikit-learn"""
    
    def __init__(self):
        self.log_reg = LogisticRegression(max_iter=1000, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models = {
            'Regresión Logística': self.log_reg,
            'Random Forest': self.rf
        }
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Entrenar y evaluar todos los modelos"""
        results = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': acc,
                'predictions': y_pred
            }
            
            print(f"\n📊 {name}")
            print(f"   Precisión: {acc:.4f}")
            print(classification_report(y_test, y_pred))
        
        return results


class PyTorchMLP(nn.Module):
    """Perceptrón Multicapa con PyTorch"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2):
        super().__init__()
        layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers_list.append(nn.Linear(prev_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers_list.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Entrenar modelo PyTorch"""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PyTorchMLP(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test_t).float().mean().item()
    
    print(f"\n🧠 PyTorch MLP")
    print(f"   Precisión: {acc:.4f}")
    
    return model, acc


class LSTMModel:
    """Modelo LSTM híbrido con TensorFlow/Keras"""
    
    def __init__(self, max_length=20, vocab_size=5000, embedding_dim=64):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
        
    def build_model(self):
        """Construir modelo LSTM híbrido"""
        # Rama de texto
        text_input = tf.keras.Input(shape=(self.max_length,), name='text_input')
        x = layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length)(text_input)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        text_output = layers.Dense(16, activation='relu')(x)
        
        # Rama numérica
        num_input = tf.keras.Input(shape=(4,), name='num_input')
        y = layers.Dense(16, activation='relu')(num_input)
        y = layers.Dense(8, activation='relu')(y)
        num_output = layers.Dense(8, activation='relu')(y)
        
        # Combinar
        combined = layers.concatenate([text_output, num_output])
        z = layers.Dense(16, activation='relu')(combined)
        z = layers.Dropout(0.2)(z)
        output = layers.Dense(1, activation='sigmoid', dtype='float32')(z)
        
        self.model = tf.keras.Model(inputs=[text_input, num_input], outputs=output)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compilar modelo"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train_model(self, X_text_train, X_num_train, y_train, 
                    X_text_val, X_num_val, y_val,
                    epochs=50, batch_size=32, patience=10):
        """Entrenar modelo con early stopping"""
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            [X_text_train, X_num_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_text_val, X_num_val], y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        return history