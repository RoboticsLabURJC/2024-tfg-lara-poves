---
title: "Comportamientos: DRL"
last_modified_at: 2025-03-06T19:11:00
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
- [Sigue carril DQN](#sigue-carril-dqm)
- [Sigue carril PPO](#sigue-carril-ppo)
- [Control de crucero adaptativo](#control-de-crucero-adaptativo)
- [Adelantamiento](#adelantamiento)

## Sigue carril DQN

El **espacio de observaciones** es continuo y coincide con el espacio de estados. Este espacio se normaliza en todos los entrenamientos y está compuesto por múltiples elementos, por lo que utilizamos un espacio de estados basado en un **diccionario**. Para manejar este espacio, en nuestros algoritmos emplearemos la política *MultiInputPolicy*. Los elementos del espacio de observaciones son:

- Desviación del carril
- Área del carril
- 5 puntos de la línea izquierda del carril
- 5 puntos de la línea derecha del carril
- Centro de masas

Este entorno tiene un **espacio de acciones discreto**, por lo que lo entrenaremos utilizando un algoritmo **DQN**. Para combinar las acciones de aceleración y giro, seguimos la regla de que a mayor aceleración, menor es el giro; aún no hemos incluido el freno. En total, contamos con 21 acciones disponibles:

```python
self.action_to_control = {
    0: (0.5, 0.0),
    1: (0.45, 0.01), 
    2: (0.45, -0.01),
    3: (0.4, 0.02),
    4: (0.4, -0.02),
    5: (0.4, 0.04),
    6: (0.4, -0.04),
    7: (0.4, 0.06),
    8: (0.4, -0.06),
    9: (0.4, 0.08),
    10: (0.4, -0.08),
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

###### Función de recompensa
El objetivo es que el coche circule por el centro del carril sin desviarse, manteniendo una conducción fluida y lo más rápida posible. Para lograrlo, hemos diseñado una función de recompensa basada principalmente en la desviación del carril y la velocidad actual del coche, normalizando y ponderando estos valores según sus respectivos pesos. Si el coche pierde el carril o colisiona, el episodio se detiene y se asigna una recompensa negativa:
```python
def _calculate_reward(self, error:str):
  if error == None:
      # Clip deviation and velocity
      r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
      r_vel = np.clip(self._velocity, 0.0, self._max_vel) / self._max_vel
      reward = 0.8 * r_dev + 0.2 * r_vel
  else:
      reward = -30

  return reward
```

La pérdida del carril puede ocurrir debido a que no se ha detectado una de las líneas, el área es cero o ha habido un cambio de carril, lo que indica que hemos perdido el carril anterior. Para abordar esta situación, añadimos una nueva verificación en la función step:
```python
dev_prev = self._dev
self._dev = self._camera.get_deviation()
assert abs(self._dev - dev_prev) <= 5, "Lost lane: changing lane"
```

###### Entrenamiento
Para entrenar el modelo, hemos utilizado un fixed_delta_seconds de 50ms, lo que equivale a entrenar a 20 FPS. Por lo tanto, en la fase de inferencia, necesitamos operar al menos a esta velocidad. Los entrenamientos duraron 1 día y un par de horas. Tras realizar diversas pruebas experimentales, identificamos los siguientes hiperparámetros que proporcionaron los mejores resultados:
```yaml
learning_rate: 0.0005
buffer_size: 1_000_000
batch_size: 1024
learning_starts: 0
gamma: 0.85
target_update_interval: 1024
train_freq: 256
gradient_steps: 2 
exploration_fraction: 0.8
exploration_final_eps: 0.0
n_timesteps: 8_000_000 
```

En la gráfica siguiente, se puede observar cómo el modelo finalmente converge:
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneDiscrete/train.png" alt="">
</figure>

#### Vídeo en un circuito visto durante el entrenamiento
<iframe width="560" height="315" src="https://www.youtube.com/embed/rzy2Vg57zA8?si=K4kRnSEYLguv1Q4T" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### Video en un circuito no visto durante el entrenamiento
<iframe width="560" height="315" src="https://www.youtube.com/embed/QT0PQfs9-m8?si=IZeDuQfLjTQYj3tt" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Sigue carril PPO

Como hemos comprobado con DQN, el espacio de acciones está restringido. Para conseguir un mejor comportamiento y mayor velocidad, debemos utilizar un algoritmo que permita un **espacio de acciones continuo**, como **PPO**. Seguiremos controlando dos elementos: el acelerador y el giro. Permitimos el rango completo para el acelerador (0-1) y limitamos el rango del giro a (-0.2, 0.2):
```python
self._max_steer = 0.2
self.action_space = spaces.Box(
    low=np.array([0.0, -self._max_steer]),
    high=np.array([1.0, self._max_steer]),
    shape=(2,),
    dtype=np.float64
)
```

Para los entrenamientos seguimos utilizando un fixed delta de 50ms (20 FPS), pero hemos introducido cambios en las observaciones:
- Hemos añadido la velocidad actual del vehículo al espacio de observaciones para que pueda entender mejor la función recompensa.
- En lugar de 5 puntos de cada línea del carril, ahora son 10.
<figure class="align-center" style="max-width: 100%"> <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneContinuous/lane10.png" alt=""> </figure>

La función recompensa sigue el siguiente esquema:

1. **Comprobación de errores**  
  Primero se verifica si ha ocurrido un error, como pérdida del carril o choque contra elementos de la carretera. Si ocurre, se finaliza el episodio y se asigna una recompensa negativa.

2. **Normalización lineal de elementos**  
  Si no hay errores, se normalizan los elementos de los que depende la recompensa:
  - **Desviación**:  
      - Se limita el valor de la desviación al rango [-100, 100].  
      - Se normaliza inversamente, donde una desviación de 0 tiene la mayor recompensa.  
  - **Giro**:  
      - Para giros bruscos, la recompensa es nula.  
      - Para giros no bruscos, se normaliza inversamente considerando el rango [-0.14, 0.14], menores giros mayor recompensa.  
  - **Acelerador**:  
      - Si las aceleraciones son bruscas [0.6, 1.0], la recompensa es nula.  
      - Para aceleraciones no bruscas [0.0, 0.6):  
          - Si se supera la velocidad máxima, se normaliza inversamente (aceleración 0 tiene mayor recompensa).  
          - En caso contrario, se normaliza de forma que mayores aceleraciones otorgan mayor recompensa.  

3. **Asignación de pesos a los elementos**  
  Finalmente, se define el peso de cada elemento en la recompensa según los siguientes criterios:
  - Si el giro o el acelerador son bruscos, tienen un peso muy elevado en la recompensa, mientras que los otros dos elementos tienen un peso mucho menor.
  - Si se supera la velocidad máxima, el acelerador tiene mayor peso para intentar frenar, penalizando también el giro debido a la alta velocidad.
  - Si el acelerador es bajo [0.0, 0.5), se penalizan menos los giros grandes, facilitando las curvas al reducir la velocidad.
  - Si el acelerador está en un rango alto [0.5, 0.6), se penalizan más los giros bruscos, enfocándose en zonas rectas o con giros leves.

```python
if error == None:
    # Deviation normalization
    r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
    
    # Steer conversion
    limit_steer = 0.14
    if abs(self._steer) > limit_steer:
        r_steer = 0
    else:
        r_steer = (limit_steer - abs(self._steer)) / limit_steer

    # Throttle conversion
    limit_throttle = 0.6
    if self._throttle >= limit_throttle:
        r_throttle = 0
    elif self._velocity > self._max_vel:
        r_throttle = (limit_throttle - self._throttle) / limit_throttle
    else:
        r_throttle = self._throttle / limit_throttle

    # Set weights
    if r_steer == 0:
        w_dev = 0.1
        w_throttle = 0.1
        w_steer = 0.8
    elif r_throttle == 0:
        w_dev = 0.1
        w_throttle = 0.8
        w_steer = 0.1
    elif self._velocity > self._max_vel:
        w_dev = 0.1
        w_throttle = 0.65
        w_steer = 0.25
    elif self._throttle < 0.5:
        w_dev = 0.65
        w_throttle = 0.25
        w_steer = 0.1 # Lower accelerations, penalize large turns less
    else: # [0.5, 0.6) throttle
        w_dev = 0.6
        w_throttle = 0.15
        w_steer = 0.25

    reward = w_dev * r_dev + w_throttle * r_throttle + w_steer * r_steer
else:
    reward = -40

return reward
```

Desarrollamos un programa para analizar la exploración de acciones durante el entrenamiento, cuyos resultados se presentan en el siguiente histograma.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneContinuous/actions.png" alt="">
</figure>

**Parémetros de entrenamiento**

Al igual que en el entrenamiento anterior, el parámetro n_steps fue clave. Además, el batch_size también tuvo un gran impacto: inicialmente, con valores bajos no lograba converger. Por último, el coeficiente de entropía fue fundamental, ya que con valores bajos siempre aprendía las acciones límite y con valores altos no llegaba a converger.

```yaml
policy: "MultiInputPolicy"
learning_rate: 0.0001
gamma: 0.85
gae_lambda: 0.9 
n_steps: 512 # The number of steps to run for each environment per update
batch_size: 512 
ent_coef: 0.1 # β
clip_range: 0.15 
n_timesteps: 4_000_000
```

Vemos cómo finalmente el entrenamiento converge con pocos episodios y de forma óptima.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneContinuous/train.png" alt="">
</figure>

En **inferencia** obtenemos muy buenos resultados, alcanzando grandes velocidades, superiores a 20 m/s. Finalmente, el modelo elige aceleraciones solo en el rango [0.5, 0.6) combinadas con giros sutiles. De este modo, se consigue seguir el carril de forma óptima y con una conducción suave. Cabe destacar que, si eliminamos la parte del rango bajo del acelerador ([0.0, 0.5)) en la función recompensa, unificando ambos rangos en uno con los mismos pesos, el entrenamiento no logra converger, aunque al final siempre seleccione el último rango.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaLaneContinuous/inference.png" alt="">
</figure>

Este es el resultado del entrenamiento en un circuito visto durante el entrenamiento:
<iframe width="560" height="315" src="https://www.youtube.com/embed/bZNfUwP14gc?si=hkee4Qfbq3wt7cgz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

También hemos llevado a cabo pruebas en inferencia en varios circuitos que no se utilizaron durante el entrenamiento, incluso en doferentes ciudades (*Town03* y *Town06*). Estos son los resultados obtenidos:
<iframe width="560" height="315" src="https://www.youtube.com/embed/WRPLzKqJdto?si=7y2vg7OXGs7Kz-zk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Hemos evaluado si el modelo entrenado utilizando la percepción del carril basada en *ground truth* funcionaría con una configuración diferente de la cámara y empleando percepción mediante una red neuronal. Los resultados obtenidos han sido satisfactorios.
<iframe width="560" height="315" src="https://www.youtube.com/embed/8kOpXYqzIGM?si=1iFCKF17bwYpwvsy" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

También se han obtenido buenos resultados al usar otro vehículos:
<iframe width="560" height="315" src="https://www.youtube.com/embed/DRmLNmMDAms?si=ISqjmCiuIC3mMNv7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Control de crucero adaptativo 

En este entorno, el objetivo es que el coche siga el carril mientras mantiene una velocidad de crucero definida por otro vehículo que circula delante a una velocidad constante durante todo el episodio.

Hemos añadido a las observaciones **20 puntos correspondientes a la zona frontal del LiDAR**, que en publicaciones anteriores identificamos como la región **FRONT**. Si el LiDAR no proporciona suficientes mediciones, estos puntos se rellenan con su rango máximo, establecido en 20 metros, ya que más allá de esta distancia las mediciones son menos precisas y de menor calidad. En caso de haber más puntos de los necesarios, se seleccionan de forma uniforme, asegurando una distribución equidistante y respetando el orden basado en la coordenada x. A continuación, se presentan ejemplos ilustrativos de estas configuraciones:
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaObstacle/laser_8m.png" alt="">
</figure>
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaObstacle/laser_13m.png" alt="">
</figure>

Para este entorno, inicialmente entrenamos un modelo capaz de seguir el carril, reutilizando los parámetros de entrenamiento óptimos previamente determinados. Basándonos en la función de recompensa utilizada anteriormente, normalizamos todas las medidas y calculamos la recompensa final asignando diferentes pesos a los valores involucrados. Cuando **no hay mediciones del láser**, los pesos son muy similares a los de la función de recompensa original. Sin embargo, al incorporar nuevas observaciones, fue necesario realizar pequeños ajustes para obtener el comportamiento deseado. En estos casos, el peso del LiDAR se establece en 0.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaPassing/train_base.png" alt="">
</figure>

Para lograr el comportamiento deseado, el modelo debe ajustar su velocidad en función del vehículo que tiene delante, evitando colisiones en todo momento. Si el coche se aproxima demasiado al vehículo delantero (**menos de 4 metros**), se considera una **colisión**, lo que resulta en la finalización del episodio con una penalización muy severa (-60). Esta penalización es mayor que la de salirse del carril (-40), ya que una colisión se considera una acción más crítica.

Para este propósito, utilizamos el modelo anterior y lo reentrenamos modificando únicamente la parte de la función de recompensa relacionada con las mediciones del LiDAR. Los mismos parámetros de entrenamiento fueron reutilizados, con la excepción del número total de pasos, que en este caso se redujo a 2_000_000 para optimizar el tiempo de entrenamiento, y el coeficiente de entropía, el cual se redujo a 0.04. Se entrenó a 10 FPS.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/follow_lane_deepRL/CarlaPassing/train.png" alt="">
</figure>

Cuando tenemos mediciones del LiDAR, el peso asignado al LiDAR y el de los demás parámetros se ajustan según la distancia al vehículo que se encuentra delante.
```python
if error == None:
      # Deviation normalization
      r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
      
      # Steer conversion
      limit_steer = 0.14
      if abs(self._steer) > limit_steer:
          r_steer = 0
      else:
          r_steer = (limit_steer - abs(self._steer)) / limit_steer
      
      # Throttle conversion
      limit_throttle = 0.6
      if self._throttle >= limit_throttle:
          r_throttle = 0
      elif self._velocity > self._max_vel:
          r_throttle = (limit_throttle - self._throttle) / limit_throttle
      else:
          r_throttle = self._throttle / limit_throttle

      # LiDAR conversion
      if self._passing and not np.isnan(self._dist_laser):
          r_laser = np.clip(self._dist_laser, MIN_DIST_LASER, MAX_DIST_LASER) - MIN_DIST_LASER
          r_laser /= (MAX_DIST_LASER - MIN_DIST_LASER)       
      else:
          r_laser = 0
    
      # Set weights
      if r_steer == 0:
          w_dev = 0.1
          w_throttle = 0.1
          w_steer = 0.8
          w_laser = 0.0
      elif r_throttle == 0:
          w_dev = 0.1
          w_throttle = 0.8
          w_steer = 0.1
          w_laser = 0.0
      elif self._velocity > self._max_vel:
          w_dev = 0.1
          w_throttle = 0.65
          w_steer = 0.25
          w_laser = 0.0
      elif r_laser != 0:
          if self._dist_laser <= 10:
              w_laser = 0.9
              w_steer = 0.0
              w_dev = 0.1
              w_throttle = 0.0
          elif self._dist_laser <= 12:
              w_laser = 0.5
              w_dev = 0.45
              w_steer = 0.05
              w_throttle = 0.0
          else:
              w_laser = 0.4
              w_dev = 0.5
              w_throttle = 0.05
              w_steer = 0.05
      else:
          w_dev = 0.6
          w_throttle = 0.2
          w_steer = 0.2
          w_laser = 0.0

      reward = w_dev * r_dev + w_throttle * r_throttle + w_steer * r_steer + w_laser * r_laser
  else:
      if "Distance" in error:
          reward = -60
      else:
          reward = -40

  return reward
```

Estos son los videos que ilustran los comportamientos logrados:

- Circuito visto durante el entrenamiento a bajas velocidades (5 m/s):
<iframe width="560" height="315" src="https://www.youtube.com/embed/Gvh9ZS0Sizc?si=0MzoxYPocButGyLd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

- Circuito visto durante el entrenamiento a altas velocidades (9 m/s):
<iframe width="560" height="315" src="https://www.youtube.com/embed/mN0Y2q6ny5w?si=MpIH6h2qMdk2pClq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

- Circuito no visto durante el entrenamiento:
<iframe width="560" height="315" src="https://www.youtube.com/embed/ieMD1wqqEaY?si=M1KdXcfQpBWqoc5z" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

- Otro vehículo delantero
<iframe width="560" height="315" src="https://www.youtube.com/embed/RmD8-avcwvU?si=QC3jXQLLI1mobcdl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Adelantamiento

Al igual que para el control de crucero adaptativo, primero hemos entrenado el modelo para seguir el carril. Las observaciones son:
- Velocidad del propio agente, es decir, del *Ego Vehicle*.
- Desviación del carril.
- Centro de masas del carril.
- Área del carril.
- 10 puntos de la línea de carril izquierda.
- 10 puntos de la línea del carril derecha.
- 10 puntos de la subzona frontal del LiDAR.
- 10 puntos de la subzona frontal derecha del LiDAR.
- 10 puntos de la subzona derecha del LiDAR.
- 10 puntos de la subzona trasera derecha del LiDAR.
- Centro de masas de la calzada.
- Área de la calzada.
- 16 puntos del límite izquierdo de la calzada.
- 16 puntos del límite derecho de la calzada.

1. *Inicio del adelantamiento: sigue-carril.*  
   El agente circula por su carril hasta que detecta con el LIDAR al vehículo que circula delante suyo (alcance de 20 metros). Durante este tramo, recibe la recompensa referente al seguimiento de carril, especificada en el código `rew_ppo_lane_overtaken`.

2. *Cambio al carril izquierdo.*  
   Una vez detectado el vehículo delantero, se inicia la maniobra de adelantamiento y la recompensa cambia para incentivar al agente a cambiar al carril de la izquierda, si lo hay, lo cual se comprueba con la red de segmentación EfficientVit. Si el vehículo delantero deja de ser percibido, se interrumpe el adelantamiento y se vuelve al modo sigue-carril.

3. *Adelantamiento: sigue-carril.*  
   Una vez el agente está en el carril izquierdo, se retoma la recompensa del seguimiento de carril hasta que el agente detecta con el LIDAR que ha sobrepasado al otro vehículo y que es seguro volver al carril inicial (9 metros).

4. *Vuelta al carril inicial.*  
   En este momento, la función de recompensa cambia para que el modelo aprenda que debe desplazarse al carril de la derecha.

5. *Fin del adelantamiento: sigue-carril.*  
   Una vez realizado el cambio de carril, la maniobra de adelantamiento ha finalizado y se reanuda la recompensa sigue-carril hasta finalizar el recorrido.

###### Diseño de Recompensas para los Cambios de Carril

Para diseñar las recompensas para los cambios de carril, hemos modificado la normalización de la desviación, permitiendo al modelo que aprenda de manera eficiente sin necesidad de indicaciones explícitas sobre cuándo y con qué magnitud realizar el giro, solamente especificamos en qué dirección debe moverse.

- Si queremos que el agente se desplace al carril de la izquierda (desviación positiva), cuanto mayor sea la desviación, mayor será la recompensa. Si la desviación es negativa, la recompensa será cero.
- Si el objetivo es volver al carril de la derecha (desviación negativa), cuanto más negativa sea la desviación, mayor será la recompensa. Si la desviación es positiva, la recompensa es nula.

Una vez normalizada la desviación, aplicamos la función *sigmoide*, expuesta en la Figura \ref{fig:sigmoide}, para acentuar la diferencia entre grandes desviaciones y aquellas más centradas. Esto permite al modelo aprender más rápido cuáles son las acciones óptimas.

Finalmente se consigue este comportamiento en inferencia:
<iframe width="560" height="315" src="https://www.youtube.com/embed/MOkeUKRlw9o?si=SSntc8PW1xanVtxb" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>