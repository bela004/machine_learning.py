import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tamanho = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
preco = np.array([150, 180, 210, 250, 270, 300, 330, 370])

modelo = LinearRegression()
modelo.fit(tamanho, preco)
previsoes = modelo.predict(tamanho)

plt.scatter(tamanho, preco, color="blue", label="Dados Reais")
plt.plot(tamanho, previsoes, color="red", label="Regressão Linear")
plt.xlabel("Tamanho da Casa (m²)")
plt.ylabel("Preço (mil R$)")
plt.title("Regressão Linear: Tamanho da Casa x Preço")
plt.legend()
plt.show()
