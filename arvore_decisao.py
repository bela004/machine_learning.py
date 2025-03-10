import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X = np.array([[20], [25], [30], [35], [40], [45], [50], [55], [60]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X, y)

plt.figure(figsize=(8, 6))
tree.plot_tree(modelo_arvore, filled=True, feature_names=["Idade"], class_names=["Não Comprou", "Comprou"])
plt.show()

