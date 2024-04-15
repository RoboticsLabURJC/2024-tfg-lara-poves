import matplotlib.pyplot as plt

# Datos para el histograma
data = [5, 15, 20, 25, 30, 35, 35, 40, 45, 50, 55, 60, 65, 70, 75]

# Tamaño del bin
bin_size = 10

# Crear el histograma
plt.hist(data, bins=range(min(data), max(data) + bin_size, bin_size), edgecolor='black')

# Etiquetas y título
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma con size_bin = 10')

# Mostrar el histograma
plt.show()


