---
title: "Autopiloto"
last_modified_at: 2024-05-22T13:31:00
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
  - [Visualización](#visualización)
  - [Zona frontal](#zona-frontal)
  - [Cálculo de estadísticas](#cálculo-de-estadísticas)
  - [Histogramas](#histogramas)

## Traffic manager

Hemos implementado una función llamada ***traffic_manager*** para controlar el tráfico de vehículos, activando el piloto automático de la lista de vehículos que recibe como entrada. Para lograrlo, es necesario activar el modo síncrono al configurar Carla.
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = delta_seconds
world.apply_settings(settings)

def traffic_manager(client:carla.Client, vehicles:list[carla.Vehicle], port:int=5000)
```

## LIDAR

Para visualizar adecuadamente los datos del láser, hemos desarrollado una nueva clase ***Lidar*** heredada de la clase *Sensor*. Al igual que en la implementación de la cámara, hemos agregado nuevos parámetros en el constructor para la visualización y sobrescrito la función *process_data()*. Esta función se encarga de visualizar el láser y actualizar las estadísticas relevantes a la zona frontal del láser, las cuales nos serán útiles para la detección de obstáculos.

```python
class Vehicle_sensors:
  def add_lidar(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, scale:int=25,
                transform:carla.Transform=carla.Transform(), front_angle:int=150, show_stats:bool=True)

class Lidar(Sensor): 
  def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor,
                scale:int, front_angle:int, yaw:float, screen:pygame.Surface, show_stats:bool)

  def process_data(self)
  def get_stat_zones(self)
  def get_meas_zones(self)
  
  def set_i_threshold(self, i:float)
  def get_i_threshold(self)
  
  def set_z_threshold(self, z:float)
  def get_z_threshold(self)
```

En primer lugar, es necesario transformar los datos del láser en una matriz de matrices, donde cada submatriz almacena las coordenadas *x*, *y*, *z* y la intensidad respectivamente. Cada una de estas submatrices representa un punto.
```python
lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
```

### Visualización

Para la representación del láser, dibujamos cada unos de estos puntos en 2D (x, y). Para mejorar la percepción visual, hemos interpolado el color de cada punto según su intensidad y el tamaño según su altura.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/interpolate.png" alt="">
</figure>

### Zona frontal

Con el fin de realizar adelantamientos, nos enfocaremos en la detección de obstáculos en la parte frontal del vehículo. Por lo tanto, examinaremos el ángulo frontal del láser, cuya amplitud es indicada por el usuario, por defecto es 150º.
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
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/three_zones.png" alt="">
</figure>

### Cálculo de estadísticas

Creamos una lista de dos elementos ***meas_zones***. El primer elemento de esta lista a su vez contiene tres listas, cada una contiene las distancias desde el punto hasta el centro del láser en el plano XY de cada zona. El segundo elemento, guarda de la misma manera las alturas *z*. Utilizamos estas medidas para calcular la media, la mediana, la desviación estándar y el mínimo en cada zona, ***stat_zones***. Actualizamos las estadísticas en cada iteración, pero, si deseamos visualizarlas en pantalla, su valor se actualiza cada segundo.

Como se puede observar en la imagen, los puntos de color rojo corresponden al propio coche, por lo tanto, hemos realizado un filtrado por intensidad para eliminarlos del cálculo estadístico. Este umbral tiene un valor predeterminado establecido en el constructor, pero hemos implementado unas funciones para consultar o modificar su valor. De manera similar, para calcular el mínimo, filtramos por altura para eliminar todos los puntos correspondientes a la calzada.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/stats.png" alt="">
</figure>

### Histogramas

Vamos a generar histogramas utilizando las distancias detectadas en la zona central frontal del vehículo, con el objetivo de distinguir la presencia de obstáculos y las distancias a las que se encuentran.

- En un escenario sin obstáculos, observamos que no hay ningún valor que sobresalga entre los demás.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_empty.png" alt="">
</figure>

- Cuando hay un coche delante, notamos cómo se dispara la columna que representa el rango de distancia de 5-6 metros, lo cual concuerda con la medida mínima detectada.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_car.png" alt="">
</figure>

- Si añadimos un camión a la izquierda, observamos que las medidas en el rango de 8-9 metros aumentan considerablemente. Hay dos columnas que sobresalen sobre las demás, indicando la presencia de dos obstáculos.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_car_truck.png" alt="">
</figure>

- En el caso de tener una moto delante, el cambio en los valores no es tan significativo como en el caso del coche, dado que es de tamaño menor. Sin embargo, el cambio es lo suficientemente notable respecto al escenario vacío como para detectar la presencia de la motocicleta.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_motorbike.png" alt="">
</figure>

Hemos creado una aplicación llamada *hist_carla.py* para recopilar los datos del láser y almacenarlos en formato csv (*hist_data.csv*). Al presionar la tecla *s*, se guardan los datos en el fichero csv, mientras que al presionar la tecla *x*, se puede cambiar la configuración en la disposición de los vehículos. Este fichero acepta el argumento *w* para sobrescribir el csv existente, *a* para añadir al final del archivo csv y *n* para no guardar los datos. Además, hemos creado un *script* para visualizar los *plots* de estos datos (*hist_plot.py*):
```bash
python3 hist_carla.py --mode w
python3 hist_plot.py 
```
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/autopilot/hist/hist_plot.png" alt="">
</figure>
