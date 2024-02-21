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
     # ----- HEADER ----- #
    st.markdown("")
    st.text("Prepared by Group 17")
    
    df = pd.read_csv(
        'earthquakes_2023_global.csv',
        usecols=range(1, 18),  # Assuming you want columns B to R (0-indexed)
        nrows=1000,
    )

    # ----- Dataset ----- #
    st.header("Dataset")
    st.write("For this clustering analysis, we will be using  “earthquake_2023_global.csv” as our main dataset.")
    st.dataframe(df)

    # ----- Objectives ----- #
    st.header("Objectives")
    st.write(
        """ 
        1. To leverage clustering techniques, specifically K-means clustering to identify high-risk regions prone to earthquakes.
        2. To develop prediction models utilising conventional machine learning methods to forecast magnitudes based on the identified high-risk regions. 
        3. To evaluate and compare the performance of the developed prediction models using appropriate evaluation metrics.

        """
    )

    # ----- Data Preprocessing ----- #
    st.header("Data Preprocessing")
    st.subheader("Handle missing & null values")
    image1 = Image.open("images/multivariate.png")
    st.image(image1, caption="Multivariate Imputation")

    st.write(
        """
        After apply multivariate imputation, missing and null values has been removed.
        """
    )

    st.subheader("Data Normalization")
    image2 = Image.open("images/minmax1.png").resize((600, 300))  # Adjust size as needed
    image3 = Image.open("images/minmax2.png").resize((600, 300))  # Adjust size as needed

    # Display the images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image2, caption="Before")

    with col2:
        st.image(image3, caption="After")

    st.write(
        """
        bla bla bla
        """
    )

    # ----- K-Means----- #
    st.header("K-Means Clustering")
    st.subheader("Elbow Method")
    image4 = Image.open("images/elbow.png").resize((600, 400))  # Adjust size as needed
    st.image(image4, caption="Elbow Plot")
    st.write(
        """
        bla bla bla
        """
    )

    st.subheader("K-Means")
    image5 = Image.open("images/kmeansscatter.png").resize((600, 400))  # Adjust size as needed
    st.image(image5, caption="K-Means")
    st.write(
        """
        bla bla bla
        """
    )
    st.subheader("Silhouette Method")

def about():
    dp_image = Image.open("image/removebgWaiee.png")
    image_column, right_column = st.columns((1,2))
    with image_column:
        st.image(dp_image, caption="")
    with right_column:
        st.subheader("Hi, I am Waiee :wave:")
        st.title("Bachelor of Computer Science (Hons.) Data Science")
        st.write("I am passionate in Data Science, AI and Machine Learning.")

        # Define the GitHub and LinkedIn links
        github_link = "[GitHub](https://github.com/waiee)"
        linkedin_link = "[LinkedIn](https://www.linkedin.com/in/waiee-zainol-9b00461ab/)"
        
        # Display the sentence with the links
        st.markdown(f"Check out my {github_link} & {linkedin_link} for more info!")



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

elif selected == "About":
    about()