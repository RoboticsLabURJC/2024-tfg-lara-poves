---
title: "Teleoperador"
last_modified_at: 2024-03-06T22:29:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
---

Durante las primeras semanas, el objetivo ha sido familiarizarse con el simulador CARLA y desarrollar un teleoperador sencillo destinado a controlar un vehículo.

## CARLA

Se ha estado investigando la ejecución de acciones básicas, como la apertura de diversos mundos, el desplazamiento del espectador y la ubicación de uno o varios vehículos con la posibilidad de elegir su modelo.

Posteriormente, se ha definido el vehículo que deseamos controlar, denominado "Ego Vehicle" en CARLA, al que hemos incorporado sensores. En esta funcionalidad, solamente se han añadido dos cámaras: una que simula la vista del conductor y otra para visualizar el vehículo en el mundo.

## Interfaz

Para la Interacción Humano-Robot (HRI), se ha utilizado la biblioteca Pygame y se ha creado una pantalla que permite visualizar el contenido de ambas cámaras de manera simultánea.

<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interface.png" alt="">
</figure>

## Control 

El teleoperador también ha sido desarrollado utilizando Pygame. En función de la tecla presionada, el vehículo recibe el correspondiente comando de control. Por ejemplo, la flecha hacia adelante se utiliza para avanzar, las teclas laterales para girar y la flecha hacia abajo para frenar.

El resultado final puede verse en el siguiente video [teleoperador](https://youtu.be/4Zh4QxjANoQ).
