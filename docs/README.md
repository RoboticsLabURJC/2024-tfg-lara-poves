# Conducción Autónoma en Entornos Urbanos enfocado en Maniobras de Adelantamiento usando Deep Reinforcement Learning

El sistema dispone de sensores LiDAR, para la detección de obstáculos en la carretera, y de cámaras RGB, para la detección del carril y segmentación de la calzada. A esta información se le aplica un post-procesado inteligente basado en modelos de DL, cuyo objetivo es extraer los datos relevantes y simplificados, que servirán como observaciones para los modelos de toma de decisiones, los cuales actuarán en el giro y el acelerador.

## Sigue carril basado en DQN

Este algoritmo permite un espacio de observaciones continuo, pero un espacio de acciones discreto. Por lo tanto, se han definido 21 acciones, combinando acelerador y giro, siguiendo la regla de que a mayor giro, menor aceleración. Las observaciones que recibe el modelo son:
- 5 puntos de cada línea del carril  
- Centro de masas del carril  
- Desviación con respecto al centro del carril  
- Área del carril  

Para la función de recompensa, se tiene en cuenta la velocidad y la desviación del coche.  
```python
reward = 0.8 * r_dev + 0.2 * r_vel
 ```

[![Ver en YouTube](https://img.youtube.com/vi/rzy2Vg57zA8/0.jpg)](https://www.youtube.com/watch?v=rzy2Vg57zA8)

## Sigue carril basado en PPO

Este algoritmo permite un espacio continuo tanto de observaciones como de acciones. El acelerador tiene un rango completo de [0.0, 1.0], mientras que el giro está limitado al intervalo [-0.2, 0.2]. A las observaciones del modelo se añade la velocidad del vehículo, lo que facilita el entendimiento de la función de recompensa.
```python
reward = r_dev * w_dev + r_throttle * w_throttle + r_steer * w_steer
```

Los resultados han sido muy satisfactorios:
- Tanto en circuitos vistos como no vistos durante los entrenamientos, incluso al cambiar la ciudad en CARLA.
- El modelo fue entrenado con una percepción perfecta basada en *ground truth*, y luego probado con un enfoque más realista basado en DL.
- Se cambió el modelo del *ego vehicle* (las dinámicas cambian) por otros modelos de vehículos, incluyendo furgonetas y motocicletas.

[![Ver en YouTube](https://img.youtube.com/vi/WRPLzKqJdto/0.jpg)](https://www.youtube.com/watch?v=WRPLzKqJdto)

## Control de crucero adaptativo basado en PPO

Se utilizó la técnica de *curriculum learning*, el modelo se entrenó inicialmente para una tarea sencilla, el seguimiento del carril, y luego se reentrenó para adaptarse a la velocidad del vehículo delantero utilizando los datos del LiDAR. Por lo tanto, se añadieron 20 puntos de la parte frontal del LiDAR a las observaciones, y también se tienen en cuenta en la función de recompensa.
```python
reward = r_dev * w_dev + r_throttle * w_throttle + r_steer * w_steer + r_lidar * w_lidar
```

Para construir el entorno de entrenamiento, se establece que el vehículo delantero circula a una velocidad constante en el rango [5, 10] durante entre 100 y 540 *steps*, ambos valores enteros y aleatorios. El modelo aprende que, a mayor velocidad, debe mantener una mayor distancia de seguridad.

[![Ver en YouTube](https://img.youtube.com/vi/mN0Y2q6ny5w/0.jpg)](https://www.youtube.com/watch?v=mN0Y2q6ny5w)

## Adelantamiento basado en PPO
