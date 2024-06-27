---
title: "Sigue carril: DRL"
last_modified_at: 2024-06-25T12:04:00
categories:
  - Blog
tags:
  - Gym
  - DQN
  - stable_baselines3
---

Con la nueva actualizacion de landau si se quiere usar pygame, antes se debe ejecutar en la terminal:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## DQN

hemos desactivado la segmentación para ganar frame spor segundo y reducir ele tiempo de entrenamiento

### CarlaDiscreteBasic
observaciones (las usamos normalizadas):
- area del carril
- 5 puntos de la linea izquierda del carril
- 5 puntos de la linea derecha del carril
- centro de masas
```python
class Camera(Sensor):     
  # reaward 
  def get_angle_lane_error(self):

  # obs
  def get_lane_area(self):
  def get_lane_cm(self):
```
Si el coche pierde el carril, el area es 0, el cm se situa en la de las esquinas, la mas cercana  a al ultima medida, y los puntos de cada linea de carril se situan en el centro de la imagen (simular que no hay carrril).

env:
  - sensor de camera driver
  - sensor de colison
  - si modo human: cmara de world también y activación de pygame

añadido angle_error: angulo de desviación en valor absoluto entre el coche y el carril
