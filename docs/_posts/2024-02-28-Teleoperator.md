---
title: "Teleoperador"
last_modified_at: 2024-03-06T15:30:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
---

En estas primeras semanas el objetivo ha sido familializarse con el simulador CARLA y desarrollar un teleoperador sencillo para controlar un vehiculo.

## CARLA

Se ha estado investigadando como realizar accione sbasicas como abrir diferentes mundos, mover al espectador o colocar en diferentes localizaciones uno o varios vehiculos eligiendo el modelo de los mismo.

Posteriormente, se ha definido el vehiculo el cual deaseamos controlar, en CARLA apodado como "Ego vehicle", al que añadiremos los sensores. En esta funcionalidad, solamente se han añadido dos camaras, una que simula la vista del conductor y otra para visualizar el vehiculo en el mundo.

## Interfaz

Para la HRI se ha usado la libreria Pygame, se ha creado una pantalla grande en la que visualizaremos el contenido de ambas camaras.

<figure class="align-center" style="width:60%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/post14/interface.png" alt="">
</figure>

## Control 

El teleoperador tambien se ha desarrollado usando Pygame, dependiendo de la tecla pulsada, el vehiculo es comando con el control correspondiente. La flecha alante es para avanzar, las laterales para girar y la de abajo para frenar.