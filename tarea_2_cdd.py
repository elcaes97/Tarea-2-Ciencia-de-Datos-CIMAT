import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier



file_path = 'bank-full.csv'

df = pd.read_csv("bank-full.csv",sep = ";")
#Primeras filas
print("\n --- Primeras filas del DataFrame ---")
print(df.head())


# ==========================================================
# Exploración inicial de la base de datos df
# ==========================================================

# 1) Revisemos los tipos de datos de cada columna
print("\n--- Tipos de datos ---")
print(df.dtypes)


# 2) Verifiquemos si existen valores faltantes en las columnas
print("\n--- Valores faltantes por columna ---")
print(df.isnull().sum())


# Nota: en este dataset los 'missing' no suelen estar como NaN,
# sino como la categoría 'unknown' en varias variables.
print("\n--- Conteo de 'unknown' por columna ---")
unknown_counts = (df.applymap(lambda x: str(x).lower() == "unknown")).sum()
print(unknown_counts[unknown_counts > 0])
unknown_counts = (df == 'unknown').sum()
unknown_percentages = (unknown_counts / len(df)) * 100
pd.DataFrame({'Count': unknown_counts, 'Percentage': unknown_percentages})


# 3) Revisemos si hay filas duplicadas
print("\n--- Número de filas duplicadas ---")
print(df.duplicated().sum())


# 4) Obtenemos un resumen estadístico de las columnas numéricas
# Identificar variables numéricas
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n--- Resumen estadístico de columnas numéricas ---")
print(f"Variables numéricas ({len(numerical_cols)}): {numerical_cols}")
print(df.describe())
# Visualizar distribuciones de las variables numéricas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols[:4]):  # Primeras 4 variables
    df[col].hist(bins=30, ax=axes[i])
    axes[i].set_title(f'Distribución de "{col}"')
    axes[i].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()


# 5) Revisamos un resumen de columnas categóricas (tipo objeto)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("y")
print("\n--- Resumen de columnas categóricas ---")
print(df.describe(include=['object']))
resumen_cat = {}
for col in categorical_cols:
    conteo = df[col].value_counts(dropna=False)
    porcentaje = df[col].value_counts(normalize=True, dropna=False) * 100
    resumen_cat[col] = pd.DataFrame({"conteo": conteo, "porcentaje": porcentaje.round(2)})

for col, tabla in resumen_cat.items():
    print(f"\n--- {col} ---")
    print(tabla)


# 6) Mostramos la dimensión de la base de datos
print("\n--- Dimensión de la base de datos (filas, columnas) ---")
print(df.shape)


# 7) Distribución de la variable respuesta 'y'
target_col='y'
target_dist = df[target_col].value_counts()
target_percent = df[target_col].value_counts(normalize=True) * 100
print("\n--- Distribución de la variable objetivo (y) ---")
print(f"Distribución de {target_col}:")
for value, count in target_dist.items():
    print(f"  {value}: {count} ({target_percent[value]:.2f}%)")



# ==========================================================
# Preprocesamiento General y análisis de correlaciones
# ==========================================================

# a) Convertimos la variable respuesta 'y' a binaria (0 = no, 1 = yes)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# b) Separamos predictores (X) y variable respuesta (y)
X = df.drop('y', axis=1)
y = df['y']  #Nota: y es diferente a target_col esta última es solo el nombre

# c) Analizamos las correlaciones entre variables
df_encoded = df.copy()
if target_col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[target_col] = le.fit_transform(df[target_col])
    
# Matriz de correlación para variables numéricas
numerical_cols_columna = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols_columna) > 1:
    corr_matrix = df_encoded[numerical_cols_columna].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
    plt.title('Matriz de Correlación - Variables Numéricas')
    plt.tight_layout()
    plt.show()
        
# Correlaciones con la variable y
if target_col in df_encoded.columns:    
    target_correlations = df_encoded[numerical_cols_columna].corrwith(df_encoded[target_col]).abs().sort_values(ascending=False)
    print("\nCorrelaciones con variable objetivo (absolutas):")
    print(target_correlations)


# d) Eliminamos espacios en blanco y pasamos a minúsculas las variables categóricas
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()

# e) Eliminamos la variable duration por falta de consistencia en la predicción.
X = X.drop(columns=['duration'])

# f) Imputación de datos: Imputar 'unknown' con la moda en job y education
for col in ["job", "education"]:
    moda = X[col].mode()[0]
    df[col] = X[col].replace("unknown", moda)



# ==========================================================
# Naive Bayes
# ==========================================================

# Diferenciar las columnas
cols_numericas = numerical_cols.copy()                                # columnas numéricas
cols_numericas.remove("duration")
cols_nominales = ['job', 'marital', 'contact', 'month', 'poutcome']        # categóricas nominales
cols_ordinales = ['education']                  # categóricas ordinales
cols_binarias = ['default','housing','loan']

# Definimos el orden en las ordinales
cat_education = [["unknown","secondary","primary","tertiary"]]
binarias = [["no", "yes"]]*len(cols_binarias)

# Codificamos
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", cols_numericas),  # dejar numéricas como están
        ("nom", OneHotEncoder(handle_unknown="ignore"), cols_nominales),
        ("ord", OrdinalEncoder(categories=cat_education), cols_ordinales),
        ("bin", OrdinalEncoder(categories = binarias), cols_binarias)
    ]
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# Separar y entrenar la base de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Clasificar
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1] #Probabilidades y=1

# ===============================
# Evaluación
# ===============================

# Validación cruzada: datos desbalanceados
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas para validación cruzada
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',  # Sensibilidad = Recall
    'precision': 'precision',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Realizar validación cruzada para cada métrica
for metric_name, metric_scoring in scoring.items():
    scores = cross_val_score(clf, X, y, cv=skf, scoring=metric_scoring)
    print(f"{metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
 
# Calcular todas las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)  # Sensibilidad = Recall
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Calcular especificidad manualmente desde la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)  # Especificidad = TN / (TN + FP)

print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Sensibilidad (Recall): {sensitivity:.4f}")
print(f"Especificidad: {specificity:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")





# ==========================================================
# LDA: Linear Discriminant Analysis
# ==========================================================

# Al igual que en Naive Bayes, identificamos las columnas igual
cols_numericas = numerical_cols.copy()                                # columnas numéricas
cols_numericas.remove("duration")
cols_nominales = ['job', 'marital', 'contact', 'month', 'poutcome']        # categóricas nominales
cols_ordinales = ['education']                  # categóricas ordinales
cols_binarias = ['default','housing','loan']

# Definimos el orden en las ordinales
cat_education = [["unknown","secondary","primary","tertiary"]]
binarias = [["no", "yes"]]*len(cols_binarias)

lda = LinearDiscriminantAnalysis()

# Codificamos
preprocessor_LDA = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), cols_numericas),  # reescalar numéricas
        ("nom", OneHotEncoder(handle_unknown="ignore"), cols_nominales),
        ("ord", OrdinalEncoder(categories=cat_education), cols_ordinales),
        ("bin", OrdinalEncoder(categories = binarias), cols_binarias)
    ]
)

clf_lda = Pipeline(steps=[
    ("preprocessor", preprocessor_LDA),
    ("classifier", lda)])

clf_lda.fit(X_train, y_train)

# Clasificar
y_pred_lda = clf_lda.predict(X_test)
y_pred_proba_lda = clf_lda.predict_proba(X_test)[:, 1] #Probabilidades y=1

# ===============================
# Evaluación
# ===============================

# Validación cruzada: datos desbalanceados
skf_lda = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas para validación cruzada
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',  # Sensibilidad = Recall
    'precision': 'precision',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Realizar validación cruzada para cada métrica
for metric_name, metric_scoring in scoring.items():
    scores = cross_val_score(clf_lda, X, y, cv=skf_lda, scoring=metric_scoring)
    print(f"{metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
 
# Calcular todas las métricas
accuracy_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda)
sensitivity_lda = recall_score(y_test, y_pred_lda)  # Sensibilidad = Recall
f1_lda = f1_score(y_test, y_pred_lda)
auc_lda = roc_auc_score(y_test, y_pred_proba_lda)

# Calcular especificidad manualmente desde la matriz de confusión
cm_lda = confusion_matrix(y_test, y_pred_lda)
tn, fp, fn, tp = cm_lda.ravel()
specificity_lda = tn / (tn + fp)  # Especificidad = TN / (TN + FP)

print(f"Exactitud (Accuracy): {accuracy_lda:.4f}")
print(f"Sensibilidad (Recall): {sensitivity_lda:.4f}")
print(f"Especificidad: {specificity_lda:.4f}")
print(f"Precisión: {precision_lda:.4f}")
print(f"F1-Score: {f1_lda:.4f}")
print(f"AUC-ROC: {auc_lda:.4f}")





# ==========================================================
# QDA: Linear Discriminant Analysis
# ==========================================================
# Al igual que en Naive Bayes, identificamos las columnas igual
cols_numericas = numerical_cols.copy()                                # columnas numéricas
cols_numericas.remove("duration")
cols_nominales = ['job', 'marital', 'contact', 'month', 'poutcome']        # categóricas nominales
cols_ordinales = ['education']                  # categóricas ordinales
cols_binarias = ['default','housing','loan']

# Definimos el orden en las ordinales
cat_education = [["unknown","secondary","primary","tertiary"]]
binarias = [["no", "yes"]]*len(cols_binarias)

qda = QuadraticDiscriminantAnalysis()

# Codificamos
preprocessor_QDA = ColumnTransformer(
    transformers=[
        ("num", "passthrough", cols_numericas),  # No requiere reescalamiento
        ("nom", OneHotEncoder(handle_unknown="ignore"), cols_nominales),
        ("ord", OrdinalEncoder(categories=cat_education), cols_ordinales),
        ("bin", OrdinalEncoder(categories = binarias), cols_binarias)
    ]
)

clf_qda = Pipeline(steps=[
    ("preprocessor", preprocessor_QDA),
    ("classifier", qda)])

clf_qda.fit(X_train, y_train)

# Clasificar
y_pred_qda = clf_qda.predict(X_test)
y_pred_proba_qda = clf_qda.predict_proba(X_test)[:, 1] #Probabilidades y=1

# ===============================
# Evaluación
# ===============================

# Validación cruzada: datos desbalanceados
skf_qda = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas para validación cruzada
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',  # Sensibilidad = Recall
    'precision': 'precision',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Realizar validación cruzada para cada métrica
for metric_name, metric_scoring in scoring.items():
    scores = cross_val_score(clf_qda, X, y, cv=skf_qda, scoring=metric_scoring)
    print(f"{metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
 
# Calcular todas las métricas
accuracy_qda = accuracy_score(y_test, y_pred_qda)
precision_qda = precision_score(y_test, y_pred_qda)
sensitivity_qda = recall_score(y_test, y_pred_qda)  # Sensibilidad = Recall
f1_qda = f1_score(y_test, y_pred_qda)
auc_qda = roc_auc_score(y_test, y_pred_proba_qda)

# Calcular especificidad manualmente desde la matriz de confusión
cm_qda = confusion_matrix(y_test, y_pred_qda)
tn, fp, fn, tp = cm_qda.ravel()
specificity_qda = tn / (tn + fp)  # Especificidad = TN / (TN + FP)

print(f"Exactitud (Accuracy): {accuracy_qda:.4f}")
print(f"Sensibilidad (Recall): {sensitivity_qda:.4f}")
print(f"Especificidad: {specificity_qda:.4f}")
print(f"Precisión: {precision_qda:.4f}")
print(f"F1-Score: {f1_qda:.4f}")
print(f"AUC-ROC: {auc_qda:.4f}")





# ==========================================================
# k-NN: k-Nearest Neighbors
# ==========================================================


# Usamos las mismas definiciones de columnas
cols_numericas = [col for col in numerical_cols if col != 'duration']
#cols_numericas = numerical_cols.copy()                                # columnas numéricas
#cols_numericas.remove("duration")
cols_nominales = ['job', 'marital', 'contact', 'month', 'poutcome']        # categóricas nominales
cols_ordinales = ['education']                  # categóricas ordinales
cols_binarias = ['default','housing','loan']

# Definimos el orden en las ordinales
cat_education = [["unknown","secondary","primary","tertiary"]]
binarias = [["no", "yes"]]*len(cols_binarias)

# Para k-NN es importante que todas las variables estén en la misma escala
knn = KNeighborsClassifier(n_neighbors=50, weights='uniform')

preprocessor_knn = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), cols_numericas),
        ("nom", OneHotEncoder(handle_unknown="ignore"), cols_nominales),
        ("ord", OrdinalEncoder(categories=cat_education), cols_ordinales),
        ("bin", OrdinalEncoder(categories=binarias), cols_binarias)
    ]
)

clf_knn = Pipeline(steps=[
    ("preprocessor", preprocessor_knn),
    ("classifier", knn)])

# Entrenar el modelo
clf_knn.fit(X_train, y_train)

# Clasificar
y_pred_knn = clf_knn.predict(X_test)
y_pred_proba_knn = clf_knn.predict_proba(X_test)[:, 1] # Probabilidades y=1

# ===============================
# Evaluación
# ===============================

# Validación cruzada: datos desbalanceados
skf_knn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas para validación cruzada
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',  # Sensibilidad = Recall
    'precision': 'precision',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Realizar validación cruzada para cada métrica
for metric_name, metric_scoring in scoring.items():
    scores = cross_val_score(clf_knn, X, y, cv=skf_knn, scoring=metric_scoring)
    print(f"{metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Calcular todas las métricas
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
sensitivity_knn = recall_score(y_test, y_pred_knn)  # Sensibilidad = Recall
f1_knn = f1_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn)

# Calcular especificidad manualmente desde la matriz de confusión
cm_knn = confusion_matrix(y_test, y_pred_knn)
tn, fp, fn, tp = cm_knn.ravel()
specificity_knn = tn / (tn + fp)  # Especificidad = TN / (TN + FP)

print(f"Exactitud (Accuracy): {accuracy_knn:.4f}")
print(f"Sensibilidad (Recall): {sensitivity_knn:.4f}")
print(f"Especificidad: {specificity_knn:.4f}")
print(f"Precisión: {precision_knn:.4f}")
print(f"F1-Score: {f1_knn:.4f}")
print(f"AUC-ROC: {auc_knn:.4f}")































