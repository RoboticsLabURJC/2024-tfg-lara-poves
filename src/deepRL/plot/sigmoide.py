import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=10, b=0.5):
    return 1 / (1 + np.exp(-a * (x - b)))

# Definir el rango de x
x = np.linspace(0, 1, 100)
y = sigmoid(x, a=10, b=0.85)

# Graficar
plt.plot(x, y, 'o-', label="sigmoid(a=10, b=0.85)")
plt.xlabel("Desviación normalizada")
plt.ylabel("Recompensa desviacoón")
plt.legend()
plt.grid()
plt.show()
