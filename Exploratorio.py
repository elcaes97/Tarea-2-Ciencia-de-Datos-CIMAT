## Preambulo ##

# Librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import funciones_para_graficar as fpg  # Módulo personalizado para graficar



def outliers_iqr(datos, column):
    """
    Esta función extraera los años de un localidad donde la medicion sea posible un outlier con IQR
    Parameters
    ----------
    datos : DataFrame de los datos de 13CVPDB.
    columna : Columna a extraer de acuerdo a site Code.
    Returns
    -------
     Devuelve los años de la columna donde el valor está fuera del rango IQR
    """
    Q1 = datos[column].quantile(0.25)
    Q3 = datos[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = datos[(datos[column] < Q1 - 1.5 * IQR) | (datos[column] > Q3 + 1.5 * IQR)]

    return outliers["Año"].tolist()


# Cargar datos
df = pd.read_csv('Datos.csv', encoding='latin-1',na_values=['NA','NA ', 'NaN', 'null', ''])

# ==========================================================
# Exploración inicial de la base de datos df
# ==========================================================
print("\n --- Primera visualización ---\n")
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
    
# Eliminación de  filas con información inecesaria 
nuevas_columnas = df.loc[2]
df_limpio =df.drop( range(12))
# Dejar como nombre de la columna año y el Site code 
df_limpio.columns =  nuevas_columnas  
df_limpio = df_limpio.rename(columns={'Site Code': 'Año' })
# Reiniciar el índice
df_limpio = df_limpio.reset_index(drop=True)

# ==========================================================
# Exploración inicial de la base de datos limpios
# ==========================================================
print("\n --- Segunda visualización ---\n")
print("Dimensiones del dataset:", df_limpio.shape)
print("\nPrimeras filas:")
print(df_limpio.head())
print("\nInformación del dataset:")
print(df_limpio.info())
# Imprimimos los nombres de las columnas que nos quedan para confirmar la limpieza.
print("\n --- Nombre de las columnas ---\n", df_limpio.columns)

#  Eliminar espacios en blanco en todo el DataFrame 
df_limpio = df_limpio.applymap(lambda x: str(x).strip() if isinstance(x, str) else x) 
#  Convertir la primera columna a enteros 
df_limpio.iloc[:, 0] = pd.to_numeric(df_limpio.iloc[:, 0], errors='coerce').astype("Int64")
#  Convertir el resto de columnas a numérico
df_limpio.iloc[:, 1:] = df_limpio.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
print("\nInformación del dataset:")
print(df_limpio.info())

#  Verifiquemos si existen valores faltantes en las columnas
print("\n--- Valores faltantes por columna ---")
print(df_limpio.isnull().sum())

# Revisemos si hay filas duplicadas
print("\n--- Número de filas duplicadas ---")
print(df_limpio.duplicated().sum())
#  Obtenemos un resumen estadístico de las columnas numéricas
print("\n--- Resumen estadístico de columnas numéricas ---")
print(df_limpio.describe())

# ==========================================================
# 2 Detección de de problemas en los datos 
# ==========================================================

# Valores faltantes 
porcentaje_nan = df_limpio.isnull().mean() * 100
print("\n---Porcentaje de valores faltantes por variable---\n")
print(porcentaje_nan.sort_values(ascending=False))

# Valores faltantes por periodo de estudio
print("\n---Porcentaje de valores faltantes por periodo de estudio---\n")
for i in range(1, 26):
# Extraer los datos de la columna i
    datos_col = df_limpio.iloc[:, i]

    # Obtener los años de inicio y fin desde df
    año_in = int(df.iloc[8, i])   
    año_fin = int(df.iloc[9, i])  
    # Calcular los índices correspondientes 
    idx_in = año_in - 1600
    idx_fin = año_fin - 1600
    # Filtrar los datos usando slicing por posición
    datos_col_filtrado = datos_col.iloc[idx_in:idx_fin + 1]

    # Calcular el porcentaje de NaN
    # Calcular estadísticas
    total_nan = datos_col_filtrado.isnull().sum()
    porcentaje_nan = datos_col_filtrado.isnull().mean() * 100
    promedio = datos_col_filtrado.mean()

    # Imprimir resultados
    print(f"Columna {df_limpio.columns[i]} ({año_in},{año_fin})")
    print(f"Numero de datos: {datos_col_filtrado.shape[0]}")
    print(f"Número de datos faltantes: {total_nan}")
    print(f"Porcentaje de faltantes: {porcentaje_nan:.2f}%")

# Posibles outliers con IQR
print("\n--- Posibles outliers con IQR ---\n ")
outlier_data = []
for n, i in enumerate(nuevas_columnas[1:]):
    outlier_anos = outliers_iqr(df_limpio, i)  # obtienes lista de años
    outlier_data.append([i, len(outlier_anos),outlier_anos])
    print(f"Outliers en {i} con IQR: {len(outlier_anos)}")
# Convertir los datos de los outliers en un arreglo  
outlier_data = np.array(outlier_data, dtype=object) 

#Posibles outliers con Z
print("\n--- Posibles outliers con Z ---\n ")
outlier_data_z = []
for i in nuevas_columnas[1:]:
    datos_col = df_limpio[i]
    media = np.mean(datos_col)
    std_dev = np.std(datos_col)
    z_scores = [(x - media) / std_dev for x in datos_col]
    outlier_anos_z = df_limpio.index[np.abs(z_scores) > 3].tolist()
    outlier_data_z.append([i, len(outlier_anos_z), outlier_anos_z])
    print(f"Outliers en {i} con Z-score: {len(outlier_anos_z)}")
# Convertir a array de objetos
outlier_data_z = np.array(outlier_data_z, dtype=object)

## Comparar el numero de posibles outliers con una graifaca de barras
# Extraer nombres de variables y cantidades de outliers
variables = [row[0] for row in outlier_data]
iqr_counts = [row[1] for row in outlier_data]
z_counts = [row[1] for row in outlier_data_z]


# Crear la figura con dos subplots (axes)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharex=True)
# Subplot 1 - IQR
ax1.bar(variables, iqr_counts, color='skyblue')
ax1.set_title('Número de Outliers por IQR')
ax1.set_ylabel('Cantidad de outliers')
ax1.grid(axis='y', linestyle='--', alpha=0.6)
# Subplot 2 - Z-score
ax2.bar(variables, z_counts, color='skyblue')
ax2.set_title('Número de Outliers por Z-score')
ax2.set_ylabel('Cantidad de outliers')
ax2.set_xlabel('Variables')
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.set_xticklabels(variables, rotation=90)

plt.tight_layout()
plt.show()



# Gráficas
fpg.graficar_todos_los_histogramas(df_limpio, nuevas_columnas)
fpg.graficar_todos_los_boxplot(df_limpio, nuevas_columnas)
fpg.graficar_todas_las_seriestiempo(df_limpio,nuevas_columnas)

# Gráficas por especie
fpg.graficar_histogramas_por_especie(df_limpio)
fpg.graficar_boxplot_por_especie(df_limpio)
fpg.graficar_seriestiempo_por_especie(df_limpio)


# ---------- 6) Quercus_petraea (FON) ----------
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(df_limpio['Año'], df_limpio["FON"], marker='o', linestyle='-', color='steelblue')
ax.set_title("FON")
ax.set_xlabel("Año")
ax.set_ylabel("13CVPDB")
fig.suptitle("Serie de tiempo de 13CVPDB en Quercus_petraea")
plt.tight_layout()
plt.show()



