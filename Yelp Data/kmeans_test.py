import numpy as np
import scipy.stats as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

business_fields = ['business_id', 'name', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'categories']


business_df = pd.read_csv("yelp_business.csv", usecols=business_fields)

print(business_df.info())

rows = np.random.choice(business_df.index.values, 75000)
sampled_df = business_df.iloc[rows]

lats = sampled_df['latitude']
longs = sampled_df['longitude']
stars = sampled_df['stars']
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(lats, longs, stars, c='r', marker='o')
#
#ax.set_xlabel('Latitude')
#ax.set_ylabel('Longitude')
#ax.set_zlabel('Stars')
#plt.show()
#
X = []
for item in zip(lats,longs,stars):
    X.append([item[0], item[1], item[2]])
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111,projection='3d')

k_means = KMeans(n_clusters=3)
k_means.fit(X)

centroids = k_means.cluster_centers_
ax.scatter(lats, longs, stars)

ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2],
            marker='x', s=300,c='green',linewidth=4)

ax.xlabel("Latitude")
ax.ylabel("Longitude")
ax.zlabel("Stars")
plt.show()
