---
title: "Autopiloto"
last_modified_at: 2024-03-14T21:01:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
  - LIDAR
  - Traffic manager
---

Una vez habituados con las funciones básicas de CARLA y realizado el teleoperador, continuamos explorando otras funcionalidades proporcionadas por CARLA que necesitaremos posteriormente.

## Traffic manager

Hemos implementado una función llamada ***traffic_manager*** para controlar el tráfico de vehículos de manera eficiente. Esta función activa el piloto automático de la lista de vehículos que recibe como entrada. Además, posibilita la modificación de ciertos parámetros de conducción, como el porcentaje de velocidad con respecto al límite máximo permitido y la distancia de seguridad entre vehículos.
```python
def traffic_manager(client:carla.Client, vehicles:List[carla.Vehicle], port:int=5000, dist:float=3.0, speed_lower:float=10.0):
```

## LIDAR

Para visualizar adecuadamente los datos del láser, hemos desarrollado una nueva clase ***Lidar*** heredada de la *Sensor*. Al igual que en la implementación para la cámara, hemos agregado nuevos parámetros en el constructor para la visualización y sobrescrito la función *show_image()*.

En primer lugar, es necesario transformar los datos del láser en una matriz de matrices, donde cada submatriz almacena las coordenadas *x*, *y*, *z* y la intensidad respectivamente. Cada una de estas submatrices representa un punto.
```python
lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
```

Para mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño del punto según su altura. A continuación, se presentan varios ejemplos en diferentes situaciones:

<figure class="align-center" style="max-width: 100%">
  <figcaption style="font-size: larger">Coche delante</figcaption> 
  <img src="{{ site.url }}{{ site.baseurl }}/images/car_lidar_front.png" alt="">
</figure>

<figure class="align-center" style="max-width: 100%">
  <figcaption style="font-size: larger">Coche a un lado</figcaption> 
  <img src="{{ site.url }}{{ site.baseurl }}/images/car_lidar_side.png" alt="">
</figure>

<figure class="align-center" style="max-width: 100%">
  <figcaption style="font-size: larger">Moto delante</figcaption> 
  <img src="{{ site.url }}{{ site.baseurl }}/images/motor_lidar.png" alt="">
</figure>

## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/h7hmnZ9t0Xs?si=VqMgGGDzFtJJ-IDO" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
