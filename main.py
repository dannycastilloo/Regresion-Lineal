import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data_v2.csv', sep=';')

X = data[['habitantes', 'ingresos']]
y = data['esp_vida']

model = LinearRegression()

model.fit(X, y)

intercepto = model.intercept_
coeficientes = model.coef_

print("Intercepto:", intercepto)
print("Coeficientes:", coeficientes)