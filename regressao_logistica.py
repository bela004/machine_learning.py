import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dados
notas = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95]).reshape(-1, 1)
aprovacao = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(notas, aprovacao, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Predição e avaliação
y_pred = modelo.predict(X_test)
print(f"Acurácia do Modelo: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Salvando os coeficientes no arquivo
with open("saida.txt", "w") as f:
    f.write(f"Coeficientes: {modelo.coef_}\n")
    f.write(f"Intercepto: {modelo.intercept_}\n")

print("Resultados salvos em saida.txt")
