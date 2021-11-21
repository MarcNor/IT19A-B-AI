import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

#Representation for Distributed Bag of Words
df = pd.read_csv('../../Word2Vec2d.csv', sep=",", usecols=[0,1])
df_words = pd.read_csv('../../Word2Vec2d.csv', sep=",", usecols=[2])
print(df.head(2))

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
plt.savefig('cluster_estimation.png')
visualizer.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(df)

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
for i in range(df.shape[0]):
    plt.text(x=df.x_vector[i]+0.1,y=df.y_vector[i]+0.1,s=df_words.word[i], 
          fontdict=dict(color='red',size=10),
          bbox=dict(facecolor='yellow',alpha=0.5))
plt.savefig('cluster_with_words.png')
plt.show()

sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.savefig('cluster_with_marker.png')
plt.show()
