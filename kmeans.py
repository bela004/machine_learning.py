import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dados = np.array([[1, 2], [2, 3], [3, 4], [10, 12], [11, 13], [12, 14]])

modelo_kmeans = KMeans(n_clusters=2, random_state=42)
modelo_kmeans.fit(dados)
labels = modelo_kmeans.labels_

plt.scatter(dados[:, 0], dados[:, 1], c=labels, cmap="coolwarm")
plt.xlabel("Variável 1")
plt.ylabel("Variável 2")
plt.title("Agrupamento com K-Means")
plt.show()
