import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

st.title("Predictor de Precios de Laptops")

company = st.selectbox('Brand', data['Company'].unique())



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

if st.button('Predice el Precio'):

    ppi = None
    if touchscreen == 'Si':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Si':
        ips = 1
    else:
        ips = 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("El precio predecido de esta laptop puede ser entre " +
             "S/."+str(prediction-1000)+ " y " + "S/."+ str(prediction+1000))
