---
title: "Autopiloto"
last_modified_at: 2025-03-05T18:33:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
  - LIDAR
  - Traffic manager
---

Una vez habituados con las funciones básicas de CARLA y realizado el teleoperador, continuamos explorando otras funcionalidades proporcionadas por CARLA que necesitaremos posteriormente.

## Índice
- [Traffic manager](#traffic-manager)
- [LIDAR](#lidar)
  - [Visualización](#visualizacion)
  - [Zona frontal](#zona-frontal)
  - [Cálculo de estadísticas](#calculo-de-estadisticas)
  - [Histogramas](#histogramas)

## Traffic manager

Hemos implementado una función llamada ***traffic_manager*** para controlar el tráfico de vehículos. Esta función activa el piloto automático en los vehículos proporcionados como entrada.
```python
tm = client.get_trafficmanager(port)
for v in vehicles:
  v.set_autopilot(True, tm_port)
```

También se pueden configurar más parámetros referentes a la conducción para diseñar comportamientos específicos, como una velocidad crucero, ignorar señales de tráfico, etc.

## LIDAR

Para visualizar adecuadamente los datos del láser, hemos desarrollado una nueva clase ***Lidar*** heredada de la clase *Sensor*. Al igual que en la implementación de la cámara, hemos agregado nuevos parámetros en el constructor para la visualización y sobrescrito la función *process_data()*. Se han agregado funciones para configurar filtros del LiDAR y obtener las métricas.
```python
class Vehicle_sensors:
  def add_lidar(self, ...)

class Lidar(Sensor): 
  def __init__(self, ...)

  def process_data(self)
  
  def set_i_threshold(self, i:float)
  def get_i_threshold(self)

  def set_z_threshold(self, z_down:float=None, z_up:float=None)
  def get_z_threshold(self)
    
  def get_dist_threshold(self)
  def set_dist_threshold(self, dist:float, zone:int)
    
  def get_stat_zones(self)  
  def get_min(self, zone:int)  
  def get_mean(self, zone:int)
```

En primer lugar, es necesario transformar los datos del LiDAR en una matriz de matrices, donde cada submatriz almacena las coordenadas *x*, *y*, *z* y la intensidad respectivamente. Cada una de estas submatrices representa un punto.
```python
lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
```

### Visualización

Para la representación del LiDAR, dibujamos cada uno de estos puntos en 2D (x, y). Para mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño según su altura.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/interpolate.png" alt="">
</figure>

### Zona frontal

Con el fin de realizar adelantamientos, nos enfocaremos en la detección de obstáculos en la parte frontal del vehículo. Por lo tanto, examinaremos el ángulo frontal del LiDAR, cuya amplitud es indicada por el usuario; por defecto es 150º.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/front_angle.png" alt="">
</figure>

En primer lugar, debemos determinar los ángulos límite que delimitan esta zona frontal, teniendo en cuenta la rotación del láser *yaw*. Partimos de un supuesto *yaw = 0*, al cual sumamos el *yaw* real y finalmente lo acotamos en un rango de [-180º, 180º].
```python
angle1 = -front_angle / 2 + yaw
angle2 = front_angle / 2 + yaw
```

Como se puede observar en el siguiente dibujo, dependiendo de qué ángulo sea mayor, debemos seguir un criterio u otro para determinar si un punto pertenece o no a la zona de interés. 
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/draw_angles.jpg" alt="">
</figure>

Dividimos este ángulo frontal en **tres zonas**: la parte izquierda (***front-left***), central (***front-front***) y derecha (***front-right***), asignándoles los índices 0, 1 y 2 respectivamente. 
```python
angle1_add = angle1 + front_angle / 3
angle2_sub = angle2 - front_angle / 3

angles = [angle1, angle1_add, angle2_sub, angle2]
```
Para establecer en qué zona se encuentra cada punto, seguimos el criterio mencionado anteriormente:
```python
if angles[i] <= angles[i + 1]:
  return angles[i] <= a <= angles[i + 1]
else:
  return angle[i] <= a or a <= angle[i + 1]
```

### Cálculo de estadísticas

Creamos una lista de dos elementos ***meas_zones***. El primer elemento contiene listas con las distancias desde el punto hasta el centro del láser en el plano XY de cada zona. El segundo elemento guarda de la misma manera las alturas *z*. Utilizamos estas medidas para calcular estadísticas como la media, la mediana, la desviación estándar y el mínimo en cada zona.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/stats.png" alt="">
</figure>

### Histogramas

Generamos histogramas utilizando las distancias detectadas en la zona central frontal del vehículo, con el objetivo de distinguir la presencia de obstáculos y las distancias a las que se encuentran.
```bash
python3 carla_hist.py --mode w
python3 plot.py 
```
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_plot.png" alt="">
</figure>
