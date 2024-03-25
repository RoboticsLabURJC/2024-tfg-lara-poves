---
title: "Autopiloto"
last_modified_at: 2024-03-25T21:00:00
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

Para visualizar adecuadamente los datos del láser, hemos desarrollado una nueva clase ***Lidar*** heredada de la clase *Sensor*. Al igual que en la implementación para la cámara, hemos agregado nuevos parámetros en el constructor para la visualización y sobrescrito la función *process_data()*. Esta función se encarga de visualizar el láser y actualizar las estadísticas relevantes a la zona frontal del láser, las cuales nos serán útiles para la detección de obstáculos.

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

Para la representación del láser dibujaremos cada unos de estos puntos en 2D (x, y). Para mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño del punto según su altura.

Para representar el láser, graficaremos cada uno de sus puntos en un plano 2D con coordenadas *x*, *y*. Con el fin de mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño del punto según su altura.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/interpolate.png" alt="">
</figure>

### Zona frontal
---

Con el fin de realizar adelantamientos, nos enfocaremos en la detección de obstáculos en la parte frontal del vehículo. Por lo tanto, examinaremos el ángulo frontal del láser, cuya amplitud es indicada por el usuario, por defecto 150º.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/front_angle.png" alt="">
</figure>

En primer lugar, debemos determinar los ángulos límite que delimitan esta zona frontal, teniendo en cuenta la rotación del láser *yaw*. Partimos de un supuesto *yaw = 0*, al cual sumamos el *yaw* real y finalmente lo acotamos en un rango de [-180º, 180º].
```python
angle1 = -front_angle / 2 + yaw
angle2 = front_angle / 2 + yaw
```

Como se puede observar en el siguiente dibujo, dependiendo de que ángulo sea mayor, debemos seguir un criterio u otro para determinar si un punto pertenece o no la zona de interés. 
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/draw_angles.jpg" alt="">
</figure>

Dividimos este ángulo frontal en **tres zonas**: la parte izquierda (***front-left***), central (***front-front***) y derecha (***front-right***), asignándoles los índices 0, 1 y 2 respectivamente. Aunque ya hemos encontrado los ángulos extremos, es necesario calcular los dos ángulos intermedios que delimitan las tres zonas. Estos cuatro ángulos se almacenan en una lista *angles*, la cual es un atributo de la clase *Lidar*. 
```
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

En nuestro caso, con un *yaw* de 90º, obtendríamos los ángulos: [-165.0, -115.0, -65.0, -15.0].
<div class="image-container">
  <div class="image-wrapper">
    <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/three_zones_color.png" alt="Image 1">
  </div>
  <div class="image-wrapper">
    <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/three_zones.png" alt="Image 2">
  </div>
</div>

#### Cálculo de estadísticas
---

#### Detección de obstáculos
---

## Demo
