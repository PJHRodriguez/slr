import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Programa 1 - Optimización con Descenso del Gradiente
x1 = np.random.uniform(-1, 1) 
x2 = np.random.uniform(-1, 1)
f_values = []  # Lista para almacenar los valores de f(x1, x2)

b = 0
w = 0 
rate = 0.01

for i in range(100):
    gradiente_x1 = 2 * x1 * np.exp(-(x1**2 + 3 * x2 ** 2))
    gradiente_x2 = 6 * x2 * np.exp(-(x1**2 + 3 * x2 ** 2))
    
    x1 -= rate * gradiente_x1
    x2 -= rate * gradiente_x2

    iteracion_f = 10 - np.exp(-(x1**2 + 3 * x2 ** 2))
    f_values.append(iteracion_f)  # Almacenar los valores de f(x1, x2)

print("Valor mínimo:", iteracion_f)   

# Programa 2 - Regresión Lineal Simple
df = pd.read_csv('Salary_dataset.csv')

x = df['YearsExperience'].values
y = df['Salary'].values
mse = []

b = 0.0
w = 0.0

sum_b = 0
sum_w = 0
sum_mse = 0

rate = 0.01

for i in range(100):
    predy = b + w * x

    for i in range(x.size):
        sum_mse += (y[i] - predy[i])**2
    m = sum_mse / x.size
    mse.append(m)

    for i in range(x.size):
        sum_b += (y[i] - predy[i])
        sum_w += ((y[i] - predy[i]) * x[i])

    gradiente_b = (-2 / x.size) * sum_b
    gradiente_w = (-2 / x.size) * sum_w

    b -= rate * gradiente_b
    w -= rate * gradiente_w

    sum_b = 0
    sum_w = 0
    sum_mse = 0

print("Valor de 'b' en regresión lineal:", b)
print("Valor de 'w' en regresión lineal:", w)

# Gráficas
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(range(100), f_values, color='green')
plt.xlabel('Iteraciones')
plt.ylabel('f(x1, x2)')
plt.title('Evolución de f(x1, x2) con Descenso del Gradiente')

plt.subplot(1, 3, 2)
plt.xlabel('Iteración')
plt.ylabel('MSE')
plt.title('Evolución del MSE durante el Descenso del Gradiente')
plt.plot(range(100), mse, color='blue')

plt.subplot(1, 3, 3)
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.title('Regresión Lineal Simple con Optimización')
plt.scatter(x, y, color="blue", marker='o')
plt.plot(x, predy, color='red')

plt.tight_layout()
plt.show()