---
title: "Percepción"
last_modified_at: 2024-03-26T18:01:00
categories:
  - Blog
tags:
  - Redes neuronales
  - Visión artificial
  - Carla
---

## Clasificación vs detección vs segmentación

La **clasificación** implica asignar etiquetas o clases a imágenes o regiones específicas, para ello pueden utilizarse CNNs. Sin embargo, esta técnica no proporciona información sobre las ubicaciones de los objetos, simplemente responde a la pregunta de si un objeto específico está presente, por ejemplo: ¿hay un perro?
<figure class="align-center" style="max-width: 70%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/classification.jpeg" alt="">
</figure>

La **detección** es el proceso que nos permite identificar varios objetos y sus ubicaciones en una imagen, proporcionando sus *bounding boxes*. Responde a pregunta: ¿qué hay en la imagen y dónde está?. Se suele utilizar para tareas en tiempo real, un ejemplo en conducción autónoma es la detección de peatones, pues nos basta con señalar y conocer su posición en la escena mediante un cuadro delimitador.
<figure class="align-center" style="max-width: 80%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/detection.jpeg" alt="">
</figure>

La **segmentación** consiste en dividir una imagen en regiones significativas con el objetivo de identificar objetos. Esta técnica abarca dos enfoques principales: 
- La segmentación **semántica** asigna una clase a cada uno de los píxeles de la imagen.
- La segmentación de **instancias** identificar diferentes objetos individuales dentro de una imagen.
La combinación de ambos enfoques se conoce como segmentación **panóptica**.

La segmentación nos proporciona información detallada sobre los límites y regiones de cada objeto: ¿qué pixel corresponde a cada objeto? En conducción autónoma se suele utilizar para la detección de la calzada.
<figure class="align-center" style="max-width: 80%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/segmentation.jpeg" alt="">
</figure>

## Redes neuronales convolucionales y recurrentes

## SAM + EfficientVit

## Aplicación

Red de segmentación semántica.