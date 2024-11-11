import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Cargar el modelo y pipeline desde archivo
file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Cargar los datos de entrenamiento como referencia para categorías
data = pd.read_csv("traineddata.csv")

st.title("Predictor de Precios de Laptops")

# Selección de entradas para la predicción
company = st.selectbox('Marca', data['Company'].unique())
type = st.selectbox('Tipo', data['TypeName'].unique())
ram = st.selectbox('Memoria RAM(en GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('Sistema Operativo', data['OpSys'].unique())
weight = st.number_input('Peso de la Laptop')
touchscreen = st.selectbox('¿Es pantalla táctil?', ['No', 'Sí'])
ips = st.selectbox('¿Es pantalla plana?', ['No', 'Sí'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Resolución de Pantalla', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['CPU_name'].unique())
hdd = st.selectbox('HDD(en GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(en GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU(en GB)', data['Gpu brand'].unique())

if st.button('Predecir'):
    # Convertir opciones de pantalla táctil y IPS a valores numéricos
    touchscreen = 1 if touchscreen == 'Sí' else 0
    ips = 1 if ips == 'Sí' else 0

    # Calcular PPI a partir de la resolución de pantalla
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

    # Codificar las variables categóricas utilizando índices del dataset
    try:
        company = float(data['Company'].unique().tolist().index(company))
        type = float(data['TypeName'].unique().tolist().index(type))
        cpu = float(data['CPU_name'].unique().tolist().index(cpu))
        gpu = float(data['Gpu brand'].unique().tolist().index(gpu))
        os = float(data['OpSys'].unique().tolist().index(os))
    except ValueError:
        st.error("Error: Una de las categorías seleccionadas no es válida para el modelo entrenado.")
        st.stop()

    # Crear array con los datos de entrada y asegurarse de que sean flotantes
    query = np.array([company, type, ram, float(weight),
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=float)

    # Llenar valores NaN en 'query' con 0
    np.nan_to_num(query, copy=False)

    # Asegurar que query tenga la forma correcta para el modelo
    query = query.reshape(1, -1)

    # Validar el número de características esperadas por el modelo
    expected_features = rf.n_features_in_
    st.write("Forma de query:", query.shape)
    st.write("Características esperadas por el modelo:", expected_features)

    if query.shape[1] != expected_features:
        st.error(f"Error: El modelo espera {expected_features} características, pero se proporcionaron {query.shape[1]}.")
    else:
        try:
            # Predecir el precio
            prediction = int(np.exp(rf.predict(query)[0]))

            st.title(f"El precio predecido de esta laptop puede ser entre S/ {prediction - 50} y S/ {prediction + 50}")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
