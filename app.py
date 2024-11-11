import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

file1 = open('laptoppricepredictor.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

st.title("Predictor de Precios de Laptops")

company = st.selectbox('Marca', data['Company'].unique())



# type of laptop

type = st.selectbox('Tipo', data['TypeName'].unique())

# Ram present in laptop

ram = st.selectbox('Memoria RAM(en GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# os of laptop

os = st.selectbox('Sistema Operativo', data['OpSys'].unique())

# weight of laptop

weight = st.number_input('Peso de la Laptop')

# touchscreen available in laptop or not

touchscreen = st.selectbox('¿Es pantalla tactil?', ['No', 'Si'])

# IPS

ips = st.selectbox('¿Es pantalla plana?', ['No', 'Si'])

# screen size

screen_size = st.number_input('Screen Size')

# resolution of laptop

resolution = st.selectbox('Resolución de Pantalla', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu

cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd

hdd = st.selectbox('HDD(en GB)', [0, 128, 256, 512, 1024, 2048])

# ssd

ssd = st.selectbox('SSD(en GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU(en GB)', data['Gpu brand'].unique())

if st.button('Predecir'):
    # Convertimos touchscreen y ips a 1 o 0 según la selección
    touchscreen = 1 if touchscreen == 'Si' else 0
    ips = 1 if ips == 'Si' else 0

    # Procesamos resolución para obtener ppi
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

    # Codificamos las variables categóricas
    company = data['Company'].unique().tolist().index(company)
    type = data['TypeName'].unique().tolist().index(type)
    cpu = data['CPU_name'].unique().tolist().index(cpu)
    gpu = data['Gpu brand'].unique().tolist().index(gpu)
    os = data['OpSys'].unique().tolist().index(os)

    # Creamos el array con los datos de entrada y convertimos a float
    query = np.array([company, type, ram, float(weight),
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=float)

    # Aseguramos que query tenga la forma (1, n)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Expandimos `query` para que tenga 38 características llenando con ceros
    def expand_features(input_array):
        expanded_array = np.zeros((1, 38))
        expanded_array[0, :12] = input_array  # Coloca los primeros 12 valores en la posición correspondiente
        return expanded_array

    # Transformamos `query` para que tenga 38 columnas
    expanded_query = expand_features(query)

    # Realizamos la predicción y calculamos el precio final usando `expanded_query`
    prediction = int(np.exp(rf.predict(expanded_query)[0]))

    st.title("El precio predecido de esta laptop puede ser entre " +
             "S/." + str(prediction - 1000) + " y " + "S/." + str(prediction + 1000))
