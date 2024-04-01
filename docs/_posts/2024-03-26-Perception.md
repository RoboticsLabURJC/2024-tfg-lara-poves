---
title: "Percepción"
last_modified_at: 2024-03-26T18:01:00
categories:
  - Blog
tags:
  - Redes neuronales
  - Deep learning
  - CNN
  - RNN
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

## Deep learning

La **Inteligencia Artificial** es una técnica que permite a una máquina imitar comportamientos humanos. El ***Machine Learning***, o aprendizaje automático, es el método para lograr IA a través de algoritmos entrenados con datos.

El ***deep learning*** es un tipo de machine learning inspirado por la estructura del cerebro humano, con las redes neuronales como base principal. Es capaz de reconocer patrones en los datos de entrada, a diferencia del *machine learning*, al cual hay que proporcionarle cuáles son las características distintivas, por ejemplo el color para distinguir entre tomates y limones. Los desafíos del *deep learning* son la gran cantidad de datos requerida, lo cual demanda potencia computacional elevada y conlleva procesos de entrenamiento largos.

### Redes neuronales
---
El set de datos para el entrenamiento de una red neuronal se divide en tres bloques:
- *Training data*: entrenar el modelo.
- *Validating data*: evaluar el modelo durante el entrenamiento.
- *Testing data*:  evaluar el rendimiento del modelo al finalizar el entrenamiento.

Las redes neuronales pueden resolver dos tipos de problemas: clasificación (salida finita) y regresión (salida continua). Una red neuronal se divide en tres bloques: la capa de entrada, las capas ocultas y la capa de salida, cuyo número de neuronas debe ser igual al número de salidas de la red. El número de neuronas de las capas ocultas se determina mediante experimentación.

En la siguiente imagen podemos ver un ejemplo de red neuronal, en el que cada neurona de una capa está conectada a todas las neuronas de la siguiente capa, esto se conoce como *fully connected*. Estos son algunos términos que debemos conocer para entender la estructura de una red neuronal:
- w: matriz de pesos
- b: matriz de término independiente
- d: número de características de entrada
- p: número de neuronas y salidas de una capa
- f: función de activación
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/network.png" alt="">
</figure>

El proceso de entrenamiento se divide en dos etapas:
1. **Propagación hacia adelante**: de entrada a salida, cuyo objetivo es hacer una predicción.
- Combinación lineal: r = x * w + b. 
  - En el caso de ser una neurona y dos características de entrada: r = x1 * w1 + x2 * w2 + b
- Predicción: resultado al aplicar la función de activación: a = f(r)

Las **funciones de activación** pretenden introducir no linealidad en la red, las más usadas son ReLu = max(r, 0) y softmax, usada para resolver problemas de clasificación multiclase, ya que calcula la probabilidad de que un dato pertenezca a cada una de las posibles clases.

2. **Propagación hacia atrás**: de salida a entrada, cuyo objetivo es actualizar los pesos y términos independientes.
- Función de pérdida: cuantificar el error cometido, comparando la salida y la predicción.
- Cálculo del gradiente de la función de pérdida respecto a los pesos de la capa de salida. 
- Actualización de pesos y términos independientes mediante un algoritmo de optimización. 

En la fase de actualización, se emplea un parámetro llamado **tasa de aprendizaje** para controlar la magnitud de los ajustes realizados en los pesos de la red neuronal durante cada paso de entrenamiento. Una tasa de aprendizaje muy grande puede provocar oscilaciones y dificultar la convergencia al punto óptimo, mientras que una tasa muy pequeña puede prolongar significativamente el tiempo de entrenamiento y el consumo de recursos computacionales

El entrenamiento se detiene cuando:
- Se ha alcanzado el número máximo de épocas indicado por el usuario.
- Se ha alcanzado la precisión deseada.
- El error de validación diverge del error de entrenamiento, lo cual significa que estamos sobreajustando la red.
<figure class="align-center" style="max-width: 80%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/error.png" alt="">
</figure>

### Redes neuronales convolucionales
---

Las redes neuronales convolucionales o **CNN** se usan para la clasificación de imágenes. Estas imágenes pueden tener dos dimensiones (filas x columnas), lo que corresponde a imágenes en escala de grises, o tres dimensiones (filas x columnas x color), correspondientes a imágenes en RGB.

En la siguiente imagen podemos observar las diferentes capas que componen una CNN, iremos analizando cada una de ellas.
<figure class="align-center" style="max-width: 80%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/error.png" alt="">
</figure>

- **Capa de convolución**
- **Capa Pooling**
- **Capa Flatten**
- **Capa fully connected**
- **Capa de salida o clasificación**

### Redes neuronales recurrentes
---

Las redes neuronales recurrentes o **RNN**

## SAM + EfficientVit

## Aplicación

Red de segmentación semántica.