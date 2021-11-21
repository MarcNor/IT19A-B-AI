import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

#Representation for Distributed Bag of Words
df = pd.read_csv('../../doc2vec_dbow_2d.csv', sep=",", usecols=[1,2,3])
df_stars = pd.read_csv('../../doc2vec_dbow_2d.csv', sep=",", usecols=[3])
print(df.head(2))
print("-----------")
print("How are the review stars distributed over the corpus?")
print(pd.value_counts(df_stars.values.ravel()))

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
plt.savefig('cluster_estimation_dbow.png')
visualizer.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(df)

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
for i in range(df.shape[0]):
    plt.text(x=df.x_vector[i]+0.1,y=df.y_vector[i]+0.1,s=df.stars[i], 
          fontdict=dict(color='red',size=10),
          bbox=dict(facecolor='yellow',alpha=0.5))
plt.savefig('cluster_with_words_dbow.png')
plt.show()

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.savefig('cluster_with_marker_dbow.png')
plt.show()


#Representation for Distributed Memory
df = pd.read_csv('../../doc2vec_dm_2d.csv', sep=",", usecols=[1,2,3])
df_stars = pd.read_csv('../../doc2vec_dbow_2d.csv', sep=",", usecols=[3])
print(df.head(2))
print("-----------")
print("How are the review stars distributed over the corpus?")
print(pd.value_counts(df_stars.values.ravel()))

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
plt.savefig('cluster_estimation_dm.png')
visualizer.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(df)

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
for i in range(df.shape[0]):
    plt.text(x=df.x_vector[i]+0.1,y=df.y_vector[i]+0.1,s=df.stars[i], 
          fontdict=dict(color='red',size=10),
          bbox=dict(facecolor='yellow',alpha=0.5))
plt.savefig('cluster_with_words_dm.png')
plt.show()

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.savefig('cluster_with_marker_dm.png')
plt.show()
