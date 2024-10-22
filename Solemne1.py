import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

csv_data = "hotel_bookings.csv"
df = pd.read_csv(csv_data)
df.head()

# Obtener la cantidad total de registros y variables (atributos)
total_registros, total_variables = df.shape
# Imprimir los resultados
print("Cantidad total de registros:", total_registros)
print("Cantidad total de variables (atributos):", total_variables)

# Obtener los tipos de datos por cada variable (atributo)
tipos_de_datos = df.dtypes
# Imprimir los resultados
print("Tipos de datos por cada variable (atributo):")
print(tipos_de_datos)

variables_numericas = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children']
# Calcular valores estadísticos básicos para las variables seleccionadas
estadisticas_basicas = df[variables_numericas].describe()
# Imprimir los resultados
print("Valores estadísticos básicos para cinco variables numéricas:")
print(estadisticas_basicas)

#Generar histogramas para las cinco (5) variables numéricas.
for variable in variables_numericas:
    plt.figure(figsize=(8, 6))
    plt.hist(df[variable], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

#Determinar valores estadísticos básicos para cinco (5) variables categóricas: cantidad de valores por atributo, valor máximo (moda), frecuencia del valor máximo, etc.
# Seleccionar cinco variables categóricas
variables_categoricas = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment']
# Calcular valores estadísticos básicos para las variables categóricas
for variable in variables_categoricas:
    print(f"Valores estadísticos básicos para la variable '{variable}':")
    # Cantidad de valores por atributo
    cantidad_valores = df[variable].nunique()
    print(f"Cantidad de valores únicos: {cantidad_valores}")
    # Valor máximo (moda)
    valor_maximo = df[variable].mode().iloc[0]
    print(f"Valor máximo (moda): {valor_maximo}")
    # Frecuencia del valor máximo
    frecuencia_maximo = df[variable].value_counts().max()
    print(f"Frecuencia del valor máximo: {frecuencia_maximo}")
    print("------------------------------------")

# Crear gráficos de barras para las variables seleccionadas
for variable in variables_categoricas:
    plt.figure(figsize=(10, 6))
    df[variable].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Gráfico de barras para la variable "{variable}"')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

#Realizar un análisis preliminar sobre los valores mínimos y máximos de los atributos de una (1) variable categórica.
# Eliminar las variables categóricas
df_numeric = df.select_dtypes(include=['int64', 'float64'])
# Calcular la matriz de correlación
correlation_matrix = df_numeric.corr()
# Generar la gráfica de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Todas las Variables Numéricas')
plt.show()

#Elegir dos variables numéricas (por ejemplo: lead_time y arrival_date_month) y generar un gráfico que muestre la correlación de las mismas.
variable1 = 'lead_time'
variable2 = 'arrival_date_month'
# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(df[variable1], df[variable2], color='skyblue', alpha=0.5)
plt.title(f'Correlación entre {variable1} y {variable2}')
plt.xlabel(variable1)
plt.ylabel(variable2)
plt.grid(True)
plt.show()

#Generar un gráfico que visualice la cantidad de Reservas asociadas al tipo de hotel, si es Resort o City. Responder a la pregunta: ¿qué tipo de hotel tiene más reservas?
# Contar la cantidad de reservas por tipo de hotel
reservas_por_hotel = df['hotel'].value_counts()
# Generar el gráfico de barras
plt.figure(figsize=(8, 6))
reservas_por_hotel.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Cantidad de Reservas por Tipo de Hotel')
plt.xlabel('Tipo de Hotel')
plt.ylabel('Cantidad de Reservas')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
# Responder a la pregunta
tipo_hotel_mas_reservas = reservas_por_hotel.idxmax()
cantidad_reservas_max = reservas_por_hotel.max()
print(f"El tipo de hotel con más reservas es '{tipo_hotel_mas_reservas}' con {cantidad_reservas_max} reservas.")

#Preparación de los datos. Codificar, ejecutar y explicar los resultados de las instrucciones del lenguaje Python para:
#Detectar y eliminar datos duplicados:
#Identificar todos los registros duplicados de la base de datos.
duplicados = df[df.duplicated()]                                # Identificar registros duplicados
df_sin_duplicados = df.drop_duplicates()                        # Eliminar registros duplicados
csv_data_sin_duplicados = "hotel_bookings_sin_duplicados.csv"   
df_sin_duplicados.to_csv(csv_data_sin_duplicados, index=False)  # Guardar el DataFrame sin registros duplicados en un nuevo archivo CSV
print("Registros duplicados eliminados. Nuevo archivo guardado como 'hotel_bookings_sin_duplicados.csv'.")

# Identificar todas las variables (columnas) con datos faltantes
variables_con_faltantes = df.columns[df.isnull().any()]
print("Variables con datos faltantes:")
print(variables_con_faltantes)

# Elegir una variable numérica con datos faltantes y eliminar los registros (filas) que contengan dichos datos
variable_numerica_faltantes = 'lead_time'
df_sin_faltantes = df.dropna(subset=[variable_numerica_faltantes])

# Elegir una variable numérica con datos faltantes para realizar un análisis de simetría
# y determinar si es mejor la media o la mediana para sustituirlos
variable_numerica_simetria = 'children'
media = df[variable_numerica_simetria].mean()
mediana = df[variable_numerica_simetria].median()
skewness = df[variable_numerica_simetria].skew()

# Sustituir los datos faltantes por la media o la mediana dependiendo del análisis de simetría de la variable
if skewness < -1 or skewness > 1:
    df[variable_numerica_simetria].fillna(mediana, inplace=True)
    print(f"Se sustituyeron los datos faltantes en '{variable_numerica_simetria}' con la mediana.")
else:
    df[variable_numerica_simetria].fillna(media, inplace=True)
    print(f"Se sustituyeron los datos faltantes en '{variable_numerica_simetria}' con la media.")

# Elegir tres variables categóricas con datos faltantes y sustituirlos por la moda
variables_categoricas_faltantes = ['meal', 'country', 'market_segment']
for variable in variables_categoricas_faltantes:
    moda = df[variable].mode().iloc[0]
    df[variable].fillna(moda, inplace=True)
    print(f"Se sustituyeron los datos faltantes en '{variable}' con la moda: {moda}")

# Guardar el DataFrame actualizado en un nuevo archivo CSV
csv_data_actualizado = "hotel_bookings_actualizado.csv"
df.to_csv(csv_data_actualizado, index=False)
print("Datos faltantes tratados. Nuevo archivo guardado como 'hotel_bookings_actualizado.csv'.")

# Visualización de datos atípicos de la variable 'stays_in_weekend_nights' utilizando un diagrama de caja y bigotes
plt.figure(figsize=(8, 6))
df.boxplot(column='stays_in_weekend_nights')
plt.title('Diagrama de Caja y Bigotes de stays_in_weekend_nights')
plt.ylabel('Cantidad de Noches en Fin de Semana')
plt.show()

# Eliminar los registros que contengan datos atípicos de la variable 'stays_in_weekend_nights'
q1 = df['stays_in_weekend_nights'].quantile(0.25)
q3 = df['stays_in_weekend_nights'].quantile(0.75)
iqr = q3 - q1
umbral_inferior = q1 - 1.5 * iqr
umbral_superior = q3 + 1.5 * iqr
df = df[(df['stays_in_weekend_nights'] >= umbral_inferior) & (df['stays_in_weekend_nights'] <= umbral_superior)]

# Elegir tres variables numéricas diferentes y utilizar el método LOF para detectar registros con valores atípicos
variables_numericas = ['lead_time', 'stays_in_weekend_nights', 'adults']
for variable in variables_numericas:
    lof = LocalOutlierFactor()
    outliers = lof.fit_predict(df[[variable]])
    df = df[outliers == 1]

# Guardar el DataFrame actualizado en un nuevo archivo CSV
csv_data_actualizado = "hotel_bookings_actualizado.csv"
df.to_csv(csv_data_actualizado, index=False)
print("Datos atípicos tratados y eliminados. Nuevo archivo guardado como 'hotel_bookings_actualizado.csv'.")