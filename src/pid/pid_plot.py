import csv
import matplotlib.pyplot as plt

data = []
csv_files = [
    {'path': '/home/alumnos/lara/2024-tfg-lara-poves/src/pid/pid_data.csv', 'color': 'blue', 'label': 'PID'},
    {'path': '/home/alumnos/lara/2024-tfg-lara-poves/src/pid/pid_data_prev.csv', 'color': 'green', 'label': 'PID prev'}
]

for csv_file in csv_files:
    data_file = []

    with open(csv_file['path'], 'r') as file:
        lector = csv.reader(file)
        
        for row in lector:
            data_file.append(float(row[0]))
    
    # Agregar los datos del archivo actual a la lista de datos general
    data.append(data_file)

for i, data in enumerate(data):
    plt.plot(data, color=csv_files[i]['color'], label=csv_files[i]['label'])

plt.axhline(0, color='red', label='Ideal')
plt.ylim(-300, 300)
plt.legend()

plt.title('PID')
plt.xlabel('Time')
plt.ylabel('Error')

plt.show()
