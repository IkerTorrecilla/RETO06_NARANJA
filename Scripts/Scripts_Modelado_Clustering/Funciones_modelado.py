import pandas as pd
import os
from scipy.sparse import load_npz
import numpy as np
import chardet
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTETomek

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, roc_curve)

from joblib import dump, load
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -------------------------------
# Funcion para probar con y sin SMOTE Y APLICANDO PCA O SIN APLICAR PERO CON TRUNCATED SVD (MEJOR PARA SPARSE)


def preparar_datos(X, y, usar_smote=True, aplicar_pca=False, varianza_objetivo=0.9, random_state=42):
    """
    Prepara los datos para modelado.
    
    Parámetros:
    - X: matriz de features (train)
    - y: vector objetivo (train)
    - usar_smote: si True, aplica SMOTETomek al train
    - aplicar_pca: si True, aplica PCA
    - varianza_objetivo: porcentaje de varianza acumulada que queremos conservar con PCA
    - random_state: semilla para reproducibilidad
    
    Retorna:
    - X_preparado, y_preparado, pca_obj (PCA ajustado si se aplica, None si no)
    """
    
    X_prep, y_prep = X.copy(), y.copy()
    
    # Opcional: aplicar SMOTE
    if usar_smote:
        smote_tomek = SMOTETomek(random_state=random_state)
        X_prep, y_prep = smote_tomek.fit_resample(X_prep, y_prep)
        print("Shape original:", X.shape, "Shape balanceado con SMOTE:", X_prep.shape)
    else:
        print("Usando datos sin SMOTE")
    
    # Opcional: aplicar PCA
    pca_obj = None
    if aplicar_pca:
        # Ajustamos PCA con todos los componentes para calcular varianza acumulada
        pca_temp = PCA(random_state=random_state)
        pca_temp.fit(X_prep)
        cum_var = pca_temp.explained_variance_ratio_.cumsum()
        
        # Numero de componentes para alcanzar varianza objetivo
        n_comp = np.argmax(cum_var >= varianza_objetivo) + 1
        
        # Ajustamos PCA final con n_comp
        pca_obj = PCA(n_components=n_comp, random_state=random_state)
        X_prep = pca_obj.fit_transform(X_prep)
        
        print(f"PCA aplicado: {n_comp} componentes para explicar {varianza_objetivo*100:.1f}% de varianza")
    
    return X_prep, y_prep, pca_obj




# ---------------------------------------
# Evaluar un modelo de clasificación
# ---------------------------------------
def evaluar_modelo(modelo, X_test, y_test, nombre_modelo="", umbral=0.3):
    """
    Evalúa un modelo de clasificación binaria y muestra métricas y gráficas.

    Parámetros:
    ----------
    modelo : estimator o pipeline de sklearn
        Modelo entrenado (puede ser un Pipeline o un clasificador directo).
    X_test : array-like
        Conjunto de datos de test (features).
    y_test : array-like
        Etiquetas reales del conjunto de test.
    nombre_modelo : str, opcional
        Nombre del modelo (se usa en los títulos de las gráficas).
    umbral : float, opcional (default=0.3)
        Umbral de decisión para convertir probabilidades en clases (0/1).

    Retorna:
    -------
    y_prob : array
        Probabilidades estimadas para la clase positiva.
    y_pred : array
        Predicciones finales según el umbral.
    """

    clf = modelo.named_steps['clf'] if hasattr(modelo, "named_steps") else modelo


    # Si el clasificador tiene predict_proba (lo normal)
    if hasattr(clf, "predict_proba"):
        # Nos quedamos con la probabilidad de la clase positiva (columna 1)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    else:
        # Si no tiene predict_proba (ej: SVM), usamos decision_function
        y_prob = modelo.decision_function(X_test)

        # Normalizamos los valores a rango [0, 1] para poder compararlos
        den = y_prob.max() - y_prob.min()
        y_prob = (y_prob - y_prob.min()) / den if den != 0 else y_prob

    # Si la probabilidad es mayor o igual al umbral → clase 1
    # Si es menor → clase 0
    y_pred = (y_prob >= umbral).astype(int)


    acc = accuracy_score(y_test, y_pred)     
    prec = precision_score(y_test, y_pred)    
    rec = recall_score(y_test, y_pred)      
    f1 = f1_score(y_test, y_pred)             
    roc_auc = roc_auc_score(y_test, y_prob)   


    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{nombre_modelo} - Matriz de Confusión (umbral={umbral})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()


    # Calculamos FPR (false positive rate) y TPR (true positive rate)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  
    plt.title(f"{nombre_modelo} - Curva ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()


    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC AUC:   {roc_auc:.3f}")


    return y_prob, y_pred


def tabla_comparativa_modelos(modelos, X_test, y_test, umbral=0.4):
    """
    Genera una tabla comparativa con métricas de varios modelos.
    
    Parámetros:
    - modelos: dict, clave=nombre del modelo, valor=modelo entrenado
    - X_test, y_test: datos de prueba
    - umbral: float, umbral para clasificar positivos
    
    Retorna:
    - pd.DataFrame con Accuracy, Precision, Recall, F1-score y ROC AUC para cada modelo
    """
    resultados = []

    for nombre, modelo in modelos.items():
        # Obtener clasificador si es pipeline
        clf = modelo.named_steps['clf'] if hasattr(modelo, "named_steps") else modelo

        # Probabilidades o scores
        if hasattr(clf, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = modelo.decision_function(X_test)
            den = y_prob.max() - y_prob.min()
            y_prob = (y_prob - y_prob.min()) / den if den != 0 else y_prob

        # Predicciones con umbral
        y_pred = (y_prob >= umbral).astype(int)

        # Calcular mEtricas
        metrics = {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        }
        resultados.append(metrics)

    # Crear DataFrame
    tabla = pd.DataFrame(resultados)
    tabla.set_index("Modelo", inplace=True)
    return tabla

