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
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
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
    
    newdf_norm = pd.read_csv('datasets/newdf_norm.csv')
    df_clustered = pd.read_csv('datasets/df_clustered.csv')

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
        We normalize the attributes to make sure the values in same scale which is from 0 to 1.
        """
    )

    # ----- K-Means----- #
    st.header("K-Means Clustering")

    #Elbow Methods
    st.subheader("Elbow Method")
    # image4 = Image.open("images/elbow.png").resize((600, 400))  # Adjust size as needed
    # st.image(image4, caption="Elbow Plot")

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
        We apply Elbow method to find optimal number of clusters (K). Figure above visualises the sum of squared distances (SSD) for different numbers of clusters in the K-means clustering algorithm. Based on the figure, we can observe that as the number of clusters increases, the SSD decreases because more clusters allow for a better fit to the data. Therefore, we can observe that the plot shows an “elbow” or significant bend. 
        In this case, k = 2 is the optimal number of clusters. 
        """
    )

    #K-Means
    st.subheader("K-Means")
    # image5 = Image.open("images/kmeansscatter.png").resize((600, 400))  # Adjust size as needed
    # st.image(image5, caption="K-Means")

    plt.figure(figsize=(7, 3))  # Adjust size as needed
    sns.scatterplot(x="latitude", y="longitude", hue="label", data=df_clustered)
    plt.title('K-Means Clustering')
    # Display the plot in Streamlit
    st.pyplot(plt)

    st.write(
        """
        By using the scikit-learn package, we implemented K-Means clustering with the number of clusters k = 2, and random state = 1.
        Based on the figure, we can observe that the dataset has been well-clustered since there is a low amount of overlapping plots.
        """
    )

    #Silhouette Score
    st.subheader("Silhouette Method")
    image6 = Image.open("images/silhouette.png").resize((700, 400))  # Adjust size as needed
    st.image(image6, caption="Silhouette Score")
    st.write(
        """
        To evaluate our clustering performance, we have calculated the silhouette score. It gives an indication of how cohesive an object is inside its own cluster and how separated it is from other clusters. The range of the silhouette coefficient is -1 to 1. A coefficient close to +1 indicates that the sample is well-clustered and separated from neighbouring clusters, indicating that the clustering arrangement is appropriate. If the coefficient is close to zero, the sample is at the decision border between two nearby clusters. A value that's close to -1 indicates that there's a chance the sample was placed in an incorrect cluster. 

        We obtained a silhouette score of 0.7136271938207533 which indicates that clustering achieved satisfying performance and the selected features are clustered well.
        The average value silhouette score observed in the figure is between 0.7 and 0.8 silhouette coefficient values which suggests that the clusters are reasonably well-separated.

        """
    )

    #Basemap
    st.subheader("Basemap")
    image7 = Image.open("images/basemap.png").resize((700, 400))  # Adjust size as needed
    st.image(image7, caption="Basemap")

    st.write(
        """
        We have plotted the basemap using coordinates from latitude, and longitude corresponding to their labels.
        Based on the map, we can observe that there are two obvious groups of earthquake occurences locations that are clustered by K-Means.
        """
    )

    # ----- Data Post-processing----- #
    st.header("Data Post-processing")

    #Sampling
    st.subheader("Sampling")
    image8 = Image.open("images/sampling1.png").resize((600, 300))  # Adjust size as needed
    image9 = Image.open("images/sampling2.png").resize((600, 300))  # Adjust size as needed

    # Display the images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image8, caption="Before")

    with col2:
        st.image(image9, caption="After")

    st.write(
        """
        To handle imbalance the cluster label, we have applied a oversampling technique called SMOTE.
        SMOTE is suitable for numerical variables and uses KNN to find nearest neighbor, therefore we already normalized the dataset beforehand.
        Both figures above illustrate the before and after SMOTE technique applied.
        """
    )

    #Chi-Squared Test
    st.subheader("Chi-Squared Test")
    image10 = Image.open("images/kbest.png").resize((1000, 700))  # Adjust size as needed
    st.image(image10, caption="Feature Selection using Chi-Squared Test")
    st.write(
        """
        In data reduction, we have applied Chi-squared Test to select best features. 
        We have decided to select the features that have score higher than 1. Based on the result, we filtered the data and obtained features for prediction which are 
        "mag", "gap", "rms", "horizontalError", "nst", "dmin", "depth", "magNst" and the labels.

        """
    )

def pred():
    st.header("Prediction Models")

def about():
    dp_image = Image.open("images/removebgWaiee.png")
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
        options=["Clustering","Prediction Models","About"],
        index=0,
    )

if selected == "Clustering":
    st.title("Earthquake Prediction & Analysis: Clustering :bar_chart:")
    home()

elif selected == "Prediction Models":
    st.title("Earthquake Prediction & Analysis: Prediction Models :bar_chart:")
    pred()

elif selected == "About":
    st.title("About :computer:")
    about()