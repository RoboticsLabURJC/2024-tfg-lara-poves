---
title: "Sigue carril: PID"
last_modified_at: 2024-05-24T13:57:00
categories:
  - Blog
tags:
  - PID
  - Redes neuronales
---

Implementaremos una solución combinando múltiples redes neuronales para detectar el carril y calcular la desviación del vehículo respecto al mismo. Esta información se enviará a un controlador PID para corregir la desviación y mantener al vehículo en adecuadamente en el centro del carril.

## Índice
- [Detección de carril](#detección-de-carril)
  - [Red neuronal de detección de carril](#red-neuronal-de-detección-de-carril)
  - [Red neuronal de segmentación semántica](#red-neuronal-de-segmentación-semántica)
- [Controlador PID](#controlador-pid)
- [Profiling](#profiling)

## Detección de carril

Buscamos seleccionar un entorno con una única vía y rodeado de vegetación para facilitar la detección de carril.

### Red neuronal de detección de carril
Contamos con una red neuronal para detectar el carril, la cual nos proporciona dos máscaras que definen cada una de las líneas del mismo. Para mejorar la detección, especialmente en casos de líneas discontinuas o cuando la red neuronal proporciona líneas fragmentadas o incompletas, empleamos regresión lineal en los puntos obtenidos para cada línea a través de la red. Este procedimiento nos permite calcular los coeficientes de las rectas que mejor se ajustan a dichos puntos.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_pid/linear_regression.png" alt="">
</figure>

Hemos definido una altura máxima para la detección del carril, creando así la forma de un trapecio. Los puntos que se encuentren dentro de este trapecio delimitado pdefinen el área del carril. Además, hemos integrado una función de seguridad: si perdemos el seguimiento de una de las líneas del carril durante varias iteraciones consecutivas, detenemos la ejecución del programa. En casos donde no se detecten líneas, volvemos a utilizar la última medida válida.

Para filtrar mediciones erróneas, hemos implementado una memoria que guarda las cinco últimas detecciones de las líneas, junto con el ángulo que cada una forma con la horizontal. Si el ángulo de la detección actual difiere lo suficiente de la media de los ángulos almacenados en esta memoria, descartamos la medida y empleamos la última detección válida. Estas mediciones incorrectas también se consideran al evaluar si hemos perdido el carril.
<iframe width="560" height="315" src="https://www.youtube.com/embed/0MiUoJePh-s?si=tbMwHcbj9cTxUHj_" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Red neuronal de segmentación semántica
Una vez que hemos determinado el área del carril, empleamos la red de segmentación para calcular el porcentaje de ese área que corresponde realmente a la carretera. Este cálculo nos permite discernir si hemos perdido el carril aunque continuamos detectando líneas, las cuales podrían ser, por ejemplo, de la acera. Si el porcentaje de área correspondiente al carril es inferior a un umbral durante varias iteraciones seguidas, detenemos la ejecución del programa. Esto será útil sobre todo en la próxima etapa, el seguimiento del carril mediante *deep reinforcement learning*. Después de realizar todas las verificaciones, determinamos el centro de masas del carril y evaluamos su desviación con respecto al centro de la pantalla en el eje *x*, donde se encuentra nuestro vehículo.
```python
class Camera(Sensor):      
  def get_road_percentage(self)
  def get_deviation(self)
```

## Controlador PID

La desviación en el eje *x* representa el error que recibe nuestro controlador, el cual es principalmente un controlador PD para el giro del volante (*steer*). El componente proporcional normaliza el error en un rango de 0.0 a 1.0, que es el rango de control proporcionado por Carla. Sin embargo, si el error supera cierto umbral, lo incrementamos ligeramente para mejorar el rendimiento en las curvas. Respecto al componente derivativo, lo hemos incorporado para prevenir movimientos oscilatorios al salir de las curvas, ya que resta el error anterior reducido. Por lo tanto, solo consideramos el error anterior si su signo difiere del error actual, ya que de lo contrario, podría afectar negativamente la conducción en las curvas.

En cuanto al control de los pedales, si el error es menor que un umbral determinado, únicamente aceleramos; de lo contrario, aplicamos freno y reducimos la aceleración. Para lograr una velocidad alta lo antes posible, durante las primeras iteraciones incrementamos la presión sobre el acelerador.
<iframe width="560" height="315" src="https://www.youtube.com/embed/2le6gn2_RYU?si=or4z7dg026Ggcz2f" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Profiling

Hemos dividido el código en secciones para evaluar las latencias y determinar dónde estamos consumiendo más tiempo en nuestro programa, con el objetivo de mejorar su eficiencia. Como se puede observar en la siguiente imagen, la mayor parte del tiempo se destina a realizar la prediccion con el modelo de segmentación. El resto de secciones presentan latencias acordes a su carga computacional, sin presenta una desventaja significativa para nuestro programa.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_pid/profiling.png" alt="">
</figure>

Un cambio que implementamos después de este análisis fue convertir la funcionalidad que encapsula *Seg get canvas* en opcional, ya que solo es útil para la visualización, ya que únicamente modifica el color del píxel dependiendo de la clase a la que pertenezca según la red de segmentación. Con esto, logramos ganar algo de velocidad de cara al seguimiento del carril.