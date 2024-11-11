import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Cargamos los datos
data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

st.title("Predictor de Precios de Laptops")

# Selección de marca
company = st.selectbox('Marca', data['Company'].unique())

# Selección de tipo de laptop
type = st.selectbox('Tipo', data['TypeName'].unique())

# Selección de RAM
ram = st.selectbox('Memoria RAM(en GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Selección de sistema operativo
os = st.selectbox('Sistema Operativo', data['OpSys'].unique())

# Entrada de peso
weight = st.number_input('Peso de la Laptop')

# Pantalla táctil
touchscreen = st.selectbox('¿Es pantalla táctil?', ['No', 'Sí'])

# Pantalla IPS
ips = st.selectbox('¿Es pantalla plana?', ['No', 'Sí'])

# Tamaño de pantalla
screen_size = st.number_input('Screen Size')

# Resolución de pantalla
resolution = st.selectbox('Resolución de Pantalla', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Selección de CPU
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# Selección de HDD
hdd = st.selectbox('HDD(en GB)', [0, 128, 256, 512, 1024, 2048])

# Selección de SSD
ssd = st.selectbox('SSD(en GB)', [0, 8, 128, 256, 512, 1024])

# Selección de GPU
gpu = st.selectbox('GPU(en GB)', data['Gpu brand'].unique())

if st.button('Predecir'):
    # Convertimos touchscreen y ips a 1 o 0 según la selección
    touchscreen = 1 if touchscreen == 'Sí' else 0
    ips = 1 if ips == 'Sí' else 0

    # Procesamos resolución para obtener ppi
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

    # Codificamos las variables categóricas, verificando si existen en los datos entrenados
    try:
        company = data['Company'].unique().tolist().index(company)
        type = data['TypeName'].unique().tolist().index(type)
        cpu = data['CPU_name'].unique().tolist().index(cpu)
        gpu = data['Gpu brand'].unique().tolist().index(gpu)
        os = data['OpSys'].unique().tolist().index(os)
    except ValueError:
        st.error("Error: Una de las categorías seleccionadas no es válida para el modelo entrenado.")
        st.stop()  # Detenemos la ejecución si hay una categoría inválida

    # Creamos el array con los datos de entrada y convertimos a float
    query = np.array([company, type, ram, float(weight),
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=float)

    # Verificar si hay valores NaN en 'query'
    if np.isnan(query).any():
        st.error("Error: Hay valores faltantes en los datos de entrada. Por favor, verifica todas las entradas.")
        st.stop()

    # Aseguramos que query tenga la forma (1, n)
    query = query.reshape(1, -1)

    # Verificamos el número de características esperadas por el modelo
    expected_features = rf.n_features_in_

    # Agregamos las líneas de depuración
    st.write("Query shape:", query.shape)
    st.write("Model expected features:", expected_features)

    if query.shape[1] != expected_features:
        st.error(f"Error: El modelo espera {expected_features} características, pero se proporcionaron {query.shape[1]}.")
    else:
        # Realizamos la predicción y calculamos el precio final
        prediction = int(np.exp(rf.predict(query)[0]))

        st.title("El precio predecido de esta laptop puede ser entre " +
                 "S/." + str(prediction - 50) + " y " + "S/." + str(prediction + 50))
