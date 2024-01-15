# Portfolio Data Science 

# [Project 1: Linear Regression, Students Performance](https://github.com/PlayingNumbers/ball_image_classifier) 
In this project, I applied regression and multi-regression analysis to an education database. The main goal was to predict what factors influence the student's performance with higher probability. I used the sklearn.model package and seaborn for visualisation. 

Here are the main steps of my work:
- Import in pandas the libraries and the file (.csv)
- Data exploration to find the first correlation  
- Perform the 'Test, Learn & Split' test to train the model 
- Analyse the data to find outliers
- Final comments

![](/images/lr.png)


# [Project 2: K-means Segmentation](https://github.com/PlayingNumbers/ds_salary_proj) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
%matplotlib inline 

#I import the csv file in pandas in a new df named bankdf
bankdf = pd.read_csv(r"C:\Users\.....)

#I get a first overview of the data sample
bankdf.info()

#Standardise the data columns: Income & CCAvg
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_to_scale = ['Income', 'CCAvg']
data_scaled = bankdf.copy()

scaler = StandardScaler()
bankdf[['Income_scaled', 'CCAvg_scaled']] = scaler.fit_transform(bankdf[['Income','CCAvg']])
bankdf[['Income_scaled', 'CCAvg_scaled']].describe()

#Perform K-means clustering specify 3 clustering using Income & CCAvg as the features of our cluster.
#Random state: 42 
model = KMeans (n_clusters=3, random_state=42)

cluster_cols = ['Income_scaled', 'CCAvg_scaled']
model.fit(bankdf[cluster_cols])
bankdf['Cluster'] = model.predict(bankdf[cluster_cols])


# Define markers and colors
markers = ['x', '.', '_']
colors = ['blue', 'green', 'red']

# Iterate through clusters
for clust in range(3):
    temp = bankdf[bankdf.Cluster == clust]  
    plt.scatter(temp.Income, temp.CCAvg, marker=markers[clust], color=colors[clust], label='Cluster '+str(clust)) 
    
plt.xlabel('Income')
plt.ylabel('CCAvg')
plt.legend()
plt.show()

![](/images/kmeans.png)





