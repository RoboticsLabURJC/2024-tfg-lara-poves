---
title: "Sigue carril: DRL"
last_modified_at: 2024-06-19T08:35:00
categories:
  - Blog
tags:
  - Gym
  - DQN
  - stable_baselines3
---

## DQN

### CarlaDiscreteBasic
observaciones (las usamos normalizadas):
- area del carril
- 5 puntos de la linea izquierda del carril
- 5 puntos de la linea derecha del carril
- centro de masas

**Modelo 1**:
- funcion de recompensa: reward = (SIZE_CAMERA / 2 - abs(self._dev)) / (SIZE_CAMERA / 2), if out: reward = -20
- acciones de velocidad -> 3m/s
- acciones de giro -> 20 acciones -> rango [-0.2, 0.2]
- N steps: 2_000_000
- Parametros:
    model_params = {
        "learning_rate": 0.00063,
        "buffer_size": 10_000, 
        "batch_size": 128,
        "learning_starts": 0,
        "gamma": 0.5, 
        "target_update_interval": 200,
        "train_freq": 4, 
        "gradient_steps": -1,
        "exploration_fraction": 0.72, 
        "exploration_final_eps": 0.0
    }
- duracion del entrenamiento: 
