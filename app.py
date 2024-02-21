import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pydeck as pdk
import folium as folium
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Earthquake Clustering Analysis",
                   page_icon=":bar_chart:",
                   layout="wide")

def home():
    import streamlit as st
     # ----- HEADER AND OBJECTIVES ----- #
    st.markdown("")
    st.text("Prepared by Waiee Zainol")
    
    df = pd.read_excel(
        io='Significant_Earthquakes.xlsx',
        engine="openpyxl", 
        usecols='B:R',
        nrows=1000,
    )

def method():
    st.subheader("Hi, I am Waiee :wave:")

def dataanalysis():
    st.subheader("Hi, I am Waiee :wave:")

def about():
    st.subheader("Hi, I am Waiee :wave:")


# ----- SIDEBAR ----- #
with st.sidebar:
    selected = st.selectbox(
        "Menu",
        options=["Home", "Data Analysis", "Model Comparison", "About"],
        index=0,
    )

if selected == "Home":
    st.title("Earthquake Analysis: K-Means Clustering :bar_chart:")
    home()

elif selected == "Data Analysis":
    st.title("Exploratory Data Analysis")
    dataanalysis()

elif selected == "Model Comparison":
    st.title("Model Comparison")
    method()

elif selected == "About":
    about()
