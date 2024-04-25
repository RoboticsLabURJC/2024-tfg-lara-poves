---
title: "Teleoperador"
last_modified_at: 2024-04-25T13:54:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
  - Camera
  - Anaconda
---

Durante las primeras semanas, nuestro objetivo principal ha sido adentrarnos en el simulador CARLA y desarrollar un teleoperador sencillo para controlar un vehículo.

## Índice
1. [Anaconda](#anaconda)
2. [CARLA](#carla)
3. [Interfaz](#interfaz)
4. [Manejo de sensores](#manejo-de-sensores)
5. [Control](#control)
6. [Demo](#demo)

## Anaconda

Para trabajar, crearemos un entorno Anaconda que nos permita utilizar la versión deseada de Python, en este caso, utilizaremos la versión 3.7.

```bash
conda create -n tfg python=3.7
conda activate tfg
pip install pygame numpy carla==0.9.13
```

Instalamos la versión 0.9.13 de *carla*, ya que es la versión compatible con el simulador. De lo contrario, se obtienen problemas de incompatibilidad de versiones.

## CARLA

Para iniciar el simulador CARLA, usaremos el siguiente comando:
```bash
/opt/carla/CarlaUE4.sh -world-port=2000
```

Hemos estado investigando cómo realizar acciones básicas en CARLA, como la apertura de distintos entornos, el desplazamiento del observador y la colocación de uno o varios vehículos, con la opción de seleccionar su modelo.

Después, nos centramos en definir el vehículo que queríamos controlar, conocido como ***Ego Vehicle*** en CARLA, al que añadiremos los sensores. Para esta funcionalidad hemos integrado dos cámaras: una para simular la perspectiva del conductor y otra para visualizar el vehículo en su entorno.

## Interfaz

Para la Interacción Humano-Robot (HRI) hemos utilizado la biblioteca ***Pygame***, creando una pantalla que nos permite visualizar el contenido de ambas cámaras de manera simultánea.

También hemos incluido un par de botones para aumentar o disminuir el *throttle*, es decir, la cantidad de acelerador que se aplica en el coche.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/teleoperator/interface.png" alt="">
</figure>

## Manejo de sensores

Hemos creado la clase ***Vehicle_sensors***, la cual nos permite almacenar el vehículo, en nuestro caso *Ego Vehicle*, y una lista de sus sensores.
```python
class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface)

    def add_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform())    
    def add_camera_rgb(self, size_rect:tuple[int, int], init:tuple[int, int]=None, 
                       transform:carla.Transform=carla.Transform())

    def destroy(self)
```

Cada uno de los sensores pertenece a la clase ***Sensor***, la cual guarda la instancia del sensor en CARLA, contiene el *callback* que almacena los datos del sensor en una cola LIFO *thread_safe* y facilita el acceso al dato más reciente. La función ***process_data()*** debe ser implementada en cada subclase de acuerdo al tipo de sensor si deseamos procesar los datos del mismo en cada *tick*, permitiéndonos actualizar su información en la pantalla. Esta función procesa la medida del sensor almacenada en *data* con el fin de asegurar que todos los sensores operen en el mismo *frame*.

```python
class Sensor:
    def __init__(self, sensor:carla.Sensor):
        self.sensor = sensor
        self.queue = LifoQueue()
        self.sensor.listen(lambda data: self.__callback_data(data))
        self.data = None

    def __callback_data(self, data):
        self.queue.put(data)

    def update_data(self):
        self.data = self.get_last_data()

    def get_last_data(self):
        data = self.queue.get(False)
    
    def process_data(self):
        pass

class Vehicle_sensors:
    def update_data(self, flip:bool=True):
        # Pick data in the same frame
        for sensor in self.sensors:
            sensor.update_data()

        for sensor in self.sensors:
            sensor.process_data()
```

Para el manejo de la cámara, hemos desarrollado una clase ***Camera*** que hereda de *Sensor*, la cual incorpora nuevos parámetros en el constructor y sobrescribe la función *process_data()*, la cual simplemente se encarga de mostrar la imagen capturada.
```python
class Camera(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, screen:pygame.Surface)
    def process_data(self)
```
Además, hemos añadido una nueva función ***add_camera_rgb*** en la clase *Vehicle_sensors*. Esta función requiere los parámetros del constructor de la clase Camera.

## Control 

El teleoperador también ha sido desarrollado utilizando *Pygame*. En función de la tecla presionada, el vehículo recibe el correspondiente comando de control: la *w* se utiliza para avanzar, las teclas *a* y *d* para girar el volante y la *s* para frenar.

Para implementar este modo de funcionamiento, hemos creado la clase ***Teleoperator***.
```python
class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0)
    def control(self)

    def set_steer(self, steer:float)
    def set_throttle(self, throttle:float)
    def set_brake(self, brake:float)

    def get_steer(self)
    def get_throttle(self)
    def get_brake(self)
```

## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/uPWRMx6LwIM?si=qUjo2tTMJJVlc2BM&amp;start=2" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
