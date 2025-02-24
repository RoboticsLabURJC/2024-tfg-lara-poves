import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=0.1, b=10):
    return 1 / (1 + np.exp(-a * (x - b)))

# Definir el rango de x
x = np.linspace(0, 135, 400)
y = sigmoid(x, a=0.1, b=80)

# Graficar
plt.plot(x, y, 'o-', label="sigmoid(a=0.1, b=90)")
plt.xlabel("Distance")
plt.ylabel("Sigmoid Value")
plt.title("Función Sigmoide ajustada de 0 a 135 centrada en 90")
plt.legend()
plt.grid()
plt.show()
