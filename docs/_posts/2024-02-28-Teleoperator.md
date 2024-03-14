---
title: "Teleoperador"
last_modified_at: 2024-03-13T21:41:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
  - Camera
---

Durante las primeras semanas, nuestro objetivo principal ha sido adentrarnos en el simulador CARLA y desarrollar un teleoperador sencillo para controlar un vehículo.

## CARLA

Hemos estado investigando cómo realizar acciones básicas en CARLA, como la apertura de distintos entornos, el desplazamiento del observador y la colocación de uno o varios vehículos, con la opción de seleccionar su modelo.

Después, nos centramos en definir el vehículo que queríamos controlar, conocido como ***Ego Vehicle*** en CARLA, al que añadiremos los sensores. Para esta funcionalidad hemos integrado dos cámaras: una para simular la perspectiva del conductor y otra para visualizar el vehículo en su entorno.

## Interfaz

Para la Interacción Humano-Robot (HRI) hemos utilizado la biblioteca ***Pygame***, creando una pantalla que nos permite visualizar el contenido de ambas cámaras de manera simultánea.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interface.png" alt="">
</figure>

## Manejo de sensores

Hemos creado la clase ***Vehicle_sensors***, la cual nos permite almacenar el vehículo, en nuestro caso *Ego Vehicle*, y una lista de sus sensores.
```python
class Vehicle_sensors:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, screen: pygame.Surface)
    def add_sensor(self, sensor:str, size_rect:Tuple[int, int], init:Tuple[int, int]=(0, 0), transform:carla.Transform=carla.Transform())
    def update_screen(self)
    def destroy(self)
```

Cada uno de los sensores pertenece a la clase ***Sensor***, que guarda la instancia del sensor carla y contiene el *callback* que actualiza la información del sensor.
```python
class Sensor:
    def __init__(self, sensor:carla.Sensor)
    def _update_data(self, data)
    def show_image(self, screen:pygame.Surface):
      return
```

Para el manejo de la cámara, hemos desarrollado una clase ***Camera*** que hereda de *Sensor*, la cual incorpora nuevos parámetros en el constructor y sobrescribe la función *show_image()*.

## Control 

El teleoperador también ha sido desarrollado utilizando *Pygame*. En función de la tecla presionada, el vehículo recibe el correspondiente comando de control: la flecha hacia adelante se utiliza para avanzar, las teclas laterales para girar y la flecha hacia abajo para frenar.

Para implementar este modo de funcionamiento, hemos creado la clase ***Teleoperator***.
```python
class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0)
    def control(self)

    def set_steer(self, steer:float)
    def set_throttle(self, throttle:float)
    def set_brake(self, brake:float)
```

## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/4Zh4QxjANoQ?si=RHRC45ch-WrZsOHz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
