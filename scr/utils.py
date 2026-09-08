# -*- coding: utf-8 -*-
"""
Funciones auxiliares para visualización
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_distribution(y, title="Distribución de Clases", save_path=None):
    """Graficar distribución de la variable objetivo"""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette='viridis')
    plt.title(title)
    plt.xlabel('Label (0=Baja, 1=Subida)')
    plt.ylabel('Frecuencia')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión", save_path=None):
    """Graficar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_test, predictions_dict, save_path=None):
    """Graficar curvas ROC de múltiples modelos"""
    plt.figure(figsize=(10, 8))
    
    for name, y_proba in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC - Comparación de Modelos')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path=None):
    """Graficar evolución del entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Entrenamiento')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validación')
    ax1.set_title('Evolución de la Pérdida')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Entrenamiento')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validación')
    ax2.set_title('Evolución de la Precisión')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()