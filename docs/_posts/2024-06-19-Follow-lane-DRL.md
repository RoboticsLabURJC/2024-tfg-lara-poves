---
title: "Sigue carril: DRL"
last_modified_at: 2024-10-02T15:00:00
categories:
  - Blog
tags:
  - Gym
  - DQN
  - PPO
  - stable_baselines3
---

Aunque inicialmente comenzamos a realizar los entrenamientos en el antiguo servidor ***landau***, los entrenamientos definitivos se han llevado a cabo en un nuevo servidor exclusivo para nosotros ***thor***. Es importante recordar que, si se desea utilizar *pygame*, primero se debe ejecutar el siguiente comando en la terminal:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## Índice
- [Sigue carrils](#sigue-carril)
  - [CarlaLaneDiscrete](#carlalanediscrete)
  - [CarlaLaneContinuous](#carlalanecontinuous)

## Sigue carril

Para aumentar los frames por segundo y reducir el tiempo de entrenamiento, hemos desactivado la segmentación. Hemos creado tres entornos de Gym que principalmente difieren en la función de recompensa y el espacio de acciones, lo cual determina el tipo de algoritmo que utilizaremos para entrenar. Emplearemos modelos predefinidos de la librería *stable-baselines3*, como se detalló en el apartado anterior.

A continuación, se muestra el mapa del recorrido que seguirán los vehículos. Los círculos representan los distintos puntos de inicio, de los cuales se selecciona uno de manera aleatoria al comienzo de cada episodio. Cada localización está marcada con una flecha que indica la dirección del recorrido.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/map.jpg" alt="">
</figure>

El vehículo del entorno consta de los siguientes **sensores**:
- Sensor de cámara del conductor: detección de carril. Si se pierde la percepción, el episodio se interrumpe (los sensores generan una excepción que se captura).
- Sensor de colisión: si el coche se choca, detenemos el episodio.
- Si está en modo humano: cámara del entorno y activación de pygame.

El **espacio de observaciones** es continuo y coincide con el espacio de estados. Se normaliza en todos los entrenamientos y está compuesto por múltiples elementos, por lo que utilizamos un espacio de estados basado en un **diccionario**. Por consiguiente, en nuestros algoritmos emplearemos la política *MultiInputPolicy*.
- Desviación del carril
- Área del carril
- 5 puntos de la línea izquierda del carril
- 5 puntos de la línea derecha del carril
- Centro de masas

Para acceder a toda esta información, hemos tenido que crear nuevas funciones en la clase de la cámara. Si el coche pierde el carril, el área se reduce a 0, el centro de masas se coloca en la esquina más cercana a la última medida, y los puntos de cada línea del carril se sitúan en el centro de la imagen (simulando que no hay carril).
```python
class Camera(Sensor):     
  def get_lane_area(self):
  def get_lane_cm(self):
```

### CarlaLaneDiscrete

Este entorno tiene un **espacio de acciones discreto**, por lo que lo entrenaremos utilizando un algoritmo **DQN**. Para combinar las acciones de aceleración y giro, hemos seguido la regla de que a mayor aceleración, menor es el giro; aún no se hemos incluido el freno. En total, contamos con 20 acciones disponibles.
```python
self.action_to_control = {
    0: (0.5, 0.0),
    1: (0.5, 0.01), 
    2: (0.5, -0.01),
    3: (0.4, 0.02),
    4: (0.4, -0.02),
    5: (0.4, 0.04),
    6: (0.4, -0.04),
    7: (0.3, 0.06),
    8: (0.3, -0.06),
    9: (0.3, 0.08),
    10: (0.3, -0.08),
    11: (0.3, 0.1),
    12: (0.3, -0.1),
    13: (0.3, 0.12),
    14: (0.3, -0.12),
    15: (0.2, 0.14),
    16: (0.2, -0.14),
    17: (0.2, 0.16),
    18: (0.2, -0.16),
    19: (0.1, 0.18),
    20: (0.1, -0.18)
}
```

Nuestro objetivo es que el coche circule por el centro del carril sin desviarse, manteniendo una conducción fluida y lo más rápida posible. Para lograrlo, hemos diseñado una función de recompensa que se basa principalmente en la desviación del carril y en la velocidad actual del coche, normalizando y ponderando estos valores según sus respectivos pesos. Sin embargo, si el coche pierde el carril o colisiona, el episodio se detiene y se asigna una recompensa negativa.
```python
if on_lane and no_collision: 
  dev = np.clip(self._dev, -MAX_DEV, MAX_DEV)
  vel = np.clip(self._speed, 0.0, self._max_vel) 
  reward = 0.75 * (MAX_DEV - abs(dev)) / MAX_DEV + 0.25 * vel / self._max_vel
else:
  reward = -20
```

La pérdida del carril puede deberse a que no se ha detectado una de las líneas, el área es cero o ha habido un cambio de carril, lo que indica que hemos perdido el anterior. Para abordar esta situación, añadiremos una nueva verificación en la función step:
```python
dev_prev = self._dev
self._dev = self._camera.get_deviation()
assert abs(self._dev - dev_prev) <= 5, "Lost lane: changing lane"
```

Para entrenar, hemos utilizado un *fixed_delta_seconds* de 50ms, lo que equivale a entrenar a 20 FPS. Por lo tanto, en la fase de inferencia, necesitamos operar al menos a esta velocidad. Los entrenamientos tuvieron una duración de entre 17 y 19 horas. Tras realizar diversas pruebas experimentales, identificamos los hiperparámetros que proporcionaron los mejores resultados:

```yaml
learning_rate: 0.0005 
buffer_size: 10_000
batch_size: 128
gamma: 0.8 
target_update_interval: 200
train_freq: 50 
gradient_steps: 2 
```
La frecuencia de entrenamiento resultó ser un factor clave en el proceso, al principio se utilizó un valor menor, pero los modelos no lograron converger. El ratio de exploración se reduce gradualmente durante el 72.5% del entrenamiento y, a partir de ese punto, se dejan de realizar acciones aleatorias (ε=0). En la gráfica siguiente, se puede observar cómo el modelo finalmente converge:
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneDiscrete/train.png" alt="">
</figure>

En inferencia, se observa que el seguimiento del carril no es completamente fluido, especialmente en las curvas. Esto se debe a una de las limitaciones del DQN, ya que el espacio de acciones es discreto y no permite seleccionar la acción de giro óptima en cada momento.
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ta6e-msfjCc?si=1eXPQV_v_tTk9SCU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

En las siguiente gráficas, se presenta la información recopilada durante la inferencia: la aceleración comandada al coche y su velocidad actual, la desviación del carril y la recompensa obtenida, junto con histogramas que muestran las acciones de giro y aceleración escogidas. Podemos observar claramente los momentos en los que se reduce la velocidad, correspondientes a las dos curvas pronunciadas. Sin embargo, de manera general, los histogramas indican que, predominantemente, se eligen los pares de acciones más rápidos.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneDiscrete/inference.png" alt="">
</figure>

### CarlaLaneContinuous

- fixed delta mas alto porque no daba tiempo  a evaluar la accion

- aumentado el numero de puntos
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneContinuous/lane10.png" alt="">
</figure>
