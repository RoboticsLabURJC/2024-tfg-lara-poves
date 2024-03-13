---
title: "Autopiloto y LIDAR"
last_modified_at: 2024-03-13T14:01:00
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

Para poder visualizar el láser correctamente, se han tenido que modificar las clases **Vehicle_sensors** y **Sensor** añadiendo nuevos parámetros.

## Demo
