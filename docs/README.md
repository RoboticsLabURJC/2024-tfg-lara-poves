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
