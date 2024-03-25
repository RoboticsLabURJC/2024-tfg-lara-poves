---
title: "Autopiloto"
last_modified_at: 2024-03-24T21:00:00
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
1. [Traffic manager](#traffic-manager)
2. [LIDAR](#lidar)
   - [Visualización](#visualización)
   - [Zona frontal](#zona-frontal)
     - [Cálculo de estadísticas](#cálculo-de-estadísticas)
     - [Detección de obstáculos](#detección-de-obstáculos)
3. [Demo](#demo)

## Traffic manager

Hemos implementado una función llamada ***traffic_manager*** para controlar el tráfico de vehículos de manera eficiente. Esta función activa el piloto automático de la lista de vehículos que recibe como entrada. Además, posibilita la modificación de ciertos parámetros de conducción, como el porcentaje de velocidad con respecto al límite máximo permitido y la distancia de seguridad entre vehículos.
```python
def traffic_manager(client:carla.Client, vehicles:List[carla.Vehicle], port:int=5000, dist:float=3.0, speed_lower:float=10.0):
```

## LIDAR

Para visualizar adecuadamente los datos del láser, hemos desarrollado una nueva clase ***Lidar*** heredada de la *Sensor*. Al igual que en la implementación para la cámara, hemos agregado nuevos parámetros en el constructor para la visualización y sobrescrito la función *process_data()*. Esta función se encarga de visualizar el láser y actualizar las estadísticas relevantes a la zona frontal del láser, las cuales nos serán útiles para la detección de obstáculos.

```python
class Lidar(Sensor): 
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor,
                 scale:int, front_angle:int, yaw:float, screen:pygame.Surface)
    def process_data(self):
    def obstacle_front_right(self)
    def obstacle_front_left(self)
    def obstacle_front_front(self)
```

En primer lugar, es necesario transformar los datos del láser en una matriz de matrices, donde cada submatriz almacena las coordenadas *x*, *y*, *z* y la intensidad respectivamente. Cada una de estas submatrices representa un punto.
```python
lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
```

### Visualización
---

Para le representación del láser dibujaremos cada unos de etos puntos en 2D (x, y). Para mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño del punto según su altura.

Para representar el láser, graficaremos cada uno de sus puntos en un plano 2D con coordenadas *x*, *y*. Con el fin de mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño del punto según su altura.
<figure class="align-center" style="max-width: 75%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/interpolate.png" alt="">
</figure>

### Zona frontal
---

Con el fin de realizar adelantamientos, nos enfocaremos en la detección de obstáculos en la parte frontal del vehículo. Por lo tanto, examinaremos el ángulo frontal del láser, cuya amplitud es indicada por el usuario, por defecto 150º.

En primer lugar, debemos determinar los ángulos límite que delimitan esta zona frontal, teniendo en cuenta la rotación del láser *yaw*. Partiremos de un supuesto *yaw = 0*, al cual sumaremos el *yaw* real y finalmente acotaremos en un rango de [-180º, 180º].
```python
angle1 = -front_angle / 2 + yaw
angle2 = front_angle / 2 + yaw
```
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/draw_angles.png" alt="">
</figure>

Como se puede observar en el dibujo anterior, dependiendo de que ángulo sea mayor, deberemos seguir un criterio u otro para determinar si un punto pertenece o no la zona de interés. 
<figure class="align-center" style="max-width: 75%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/front_angle.png" alt="">
</figure>


[-165.0, -115.0, -65.0, -15.0]

#### Cálculo de estadísticas
---

#### Detección de obstáculos
---

## Demo
