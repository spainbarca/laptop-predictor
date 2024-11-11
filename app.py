import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv("traineddata.csv")

st.title("Predictor de Precios de Laptops")

# Inputs de usuario
company = st.selectbox('Marca', data['Company'].unique())
type = st.selectbox('Tipo', data['TypeName'].unique())
ram = st.selectbox('Memoria RAM(en GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('Sistema Operativo', data['OpSys'].unique())
weight = st.number_input('Peso de la Laptop (en Kg.)')
touchscreen = st.selectbox('¿Es pantalla tactil?', ['No', 'Si'])
ips = st.selectbox('¿Es LCD?', ['No', 'Si'])
screen_size = st.number_input('Pulgadas de la pantalla')
resolution = st.selectbox('Resolución de Pantalla', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['CPU_name'].unique())
hdd = st.selectbox('HDD(en GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(en GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU (Tarjeta Grafica)', data['Gpu brand'].unique())

# Predicción
if st.button('Predice el Precio'):
    try:
        # Conversión de entradas a numérico
        touchscreen = 1 if touchscreen == 'Si' else 0
        ips = 1 if ips == 'Si' else 0

        # Calculo de PPI
        X_resolution = int(resolution.split('x')[0])
        Y_resolution = int(resolution.split('x')[1])
        ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

        # Preparar entrada
        query = np.array([company, type, ram,os, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu]).reshape(1, -1)
        st.write("Query a predecir:", query)  # Verifica la entrada

        # Realizar predicción
        prediction = int(np.exp(rf.predict(query)[0]))
        st.title("El precio predecido de esta laptop puede ser entre " +
                 "S/."+str(prediction-50)+ " y " + "S/."+ str(prediction+50))

    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
