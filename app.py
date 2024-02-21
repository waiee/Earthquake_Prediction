import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pydeck as pdk
import folium as folium
import streamlit as st
from sklearn.cluster import KMeans
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
        'datasets/earthquakes_2023_global.csv',
        usecols=range(1, 18),  # Assuming you want columns B to R (0-indexed)
        nrows=1000,
    )

    newdf_norm = pd.read_csv('datasets/newdf_norm.csv')
    # Perform K-means clustering and calculate SSD
    max_k = 20
    ssd = []
    for i in range(1, max_k + 1):
        km_elbow = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km_elbow.fit(newdf_norm)
        ssd.append(km_elbow.inertia_)  # Sum of squared distances of samples to their closest cluster center


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
    # image4 = Image.open("images/elbow.png").resize((600, 400))  # Adjust size as needed
    # st.image(image4, caption="Elbow Plot")

    # Display the scatter elbow plot
    fig, ax = plt.subplots(figsize=(7, 3))  # Adjust size as needed
    # Plot the data
    ax.plot(range(1, max_k + 1), ssd, marker='o')
    ax.set_xticks(range(1, max_k + 1))
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('SSD')
    # Display the plot in Streamlit
    st.pyplot(fig)

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
    image6 = Image.open("images/silhouette.png").resize((600, 400))  # Adjust size as needed
    st.image(image6, caption="Silhouette Score")
    st.write(
        """
        bla bla bla
        """
    )

    st.subheader("Basemap")
    image7 = Image.open("images/basemap.png").resize((700, 400))  # Adjust size as needed
    st.image(image7, caption="Basemap")
    st.write(
        """
        bla bla bla
        """
    )

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