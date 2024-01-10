import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


import seaborn as sns 
import pickle 

#import model 
DTC_model = pickle.load(open('DTC.pkl','rb'))

#load dataset
data = pd.read_csv('Price Range Phone Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Price Range Phone')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Phone Check</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['DTC','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('List Handphone')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Price Range</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('price_range',axis=1)
y = data['price_range']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    battery_power = st.sidebar.slider('Ketahanan Baterai',0,20,1)
    blue = st.sidebar.slider('biru',0,200,108)
    clock_speed = st.sidebar.slider('kecepatan interaksi',0,140,40)
    dual_sim = st.sidebar.slider('Dual Sim',0,100,25)
    fc = st.sidebar.slider('fc',0,1000,120)
    four_g = st.sidebar.slider('4G',0,80,25)
    int_memory = st.sidebar.slider('Internal Memori', 0.05,2.5,0.45)
    m_dep = st.sidebar.slider('Jumlah Memori',21,100,24)
    mobile_wt = st.sidebar.slider('Mobile watch',21,100,24)
    n_cores = st.sidebar.slider('Number Core',21,100,24)
    px_height = st.sidebar.slider('Tinggi',21,100,24)
    px_width = st.sidebar.slider('Lebar',21,100,24)
    ramc = st.sidebar.slider('Ram',21,100,24)
    sc_w = st.sidebar.slider('sc_w',21,100,24)
    sc_h = st.sidebar.slider('sc_h',21,100,24)
    three_g = st.sidebar.slider('3G',21,100,24)
    touch_screen = st.sidebar.slider('Layar Sentuh',21,100,24)
    wifi = st.sidebar.slider('Wifi',21,100,24)

    
    user_report_data = {
        'Ketahanan Baterai':battery_power,
        'biru':blue,
        'kecepatan interaksi':clock_speed,
        'Dual Sim': dual_sim,
        'fc':fc,
        '4G':four_g,
        'Internal Memori':int_memory,
        'Jumlah Memori':m_dep,
        'Mobile watch': mobile_wt,
        'Number Core': n_cores,
        'Tinggi':px_height,
        'Lebar': px_width,
        'Ram':ramc,
        'sc_w':sc_w,
        'sc_h':sc_h,
        '3G':three_g,
        'Layar Sentuh':touch_screen,
        'Wifi':wifi,
        
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Handphone')
st.write(user_data)

user_result = DTC_model.predict(user_data)
svc_score = accuracy_score(y_test,DTC_model.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Harga Termurah'
else:
    output ='Harga Terendah'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')