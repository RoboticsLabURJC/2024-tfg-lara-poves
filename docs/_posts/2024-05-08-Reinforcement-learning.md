---
title: "Aprendizaje por refuerzo"
last_modified_at: 2024-05-08T16:51:00
categories:
  - Blog
tags:
  - Deep Reinforcement Learning
  - Q-learning
---

## Índice
1. [Tipos de entrenamientos](#tipos-de-entrenamiento)
   - [Entrenamiento supervisado](#entrenamiento-supervisado)
   - [Entrenamiento no supervisado](#entrenamiento-no-supervisado)
   - [Entrenamiento semi-supervisado](#entrenamiento-semi-supervisado)
2. [Q-learning](#q-learning)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)

## Tipos de entrenamiento 
En el contexto de aprendizaje automático, existen varios tipos de entrenamiento: supervisado, no supervisado y semi-supervisados.

<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/qlearning/training_type.png" alt="">
</figure>

### Entrenamiento supervisado
El conjunto de datos de entrenamiento es etiquetado, cada una de las entradas tiene una salida asociada a la respuesta correcta según el modelo que quiere predecir. Durante el entrenamiento, el modelo ajusta sus parámetros para minimizar el error entre las predicciones y las etiquetas reales. El objetivo de este algoritmo de aprendizaje es predecir etiquetas para datos no visto anteriormente, es ampliamente utilizado en tareas de clasificación y regresión. La precisión del modelo depende de la calidad de los datos de entrenamiento.

### Entrenamiento no supervisado
El aprendizaje no supervisado opera en conjuntos de datos sin etiquetas, donde el objetivo del modelo es identificar patrones y estructuras, agrupando los datos en categorías o clústeres. Las tareas principales incluyen el *clustering* y la reducción de dimensionalidad. Este enfoque puede ser más rentable que el aprendizaje supervisado, ya que no requiere la creación y etiquetado de grandes conjuntos de datos de entrenamiento. En este tipo de aprendizaje, la calidad de los resultados depende en gran medida de la elección adecuada del algoritmo y los parámetros utilizados.

### Entrenamiento semi-supervisado
El aprendizaje semi-supervisado emplea una combinación de datos etiquetados y no etiquetados en el conjunto de entrenamiento. Este enfoque es útil cuando el etiquetado de datos es costoso o difícil de obtener en grandes cantidades. A diferencia del aprendizaje supervisado tradicional, aquí se pueden lograr resultados significativos con solo unos pocos ejemplos etiquetados, lo que hace que el proceso sea más eficiente y práctico en ciertos escenarios. Este proceso se asemeja más a cómo aprendemos las personas.

## Q-learning

## Deep Reinforcement Learning