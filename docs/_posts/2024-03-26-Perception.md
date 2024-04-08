---
title: "Percepción"
last_modified_at: 2024-04-04T21:20:00
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
1. **Propagación hacia delante**: de entrada a salida, cuyo objetivo es hacer una predicción.
- Combinación lineal: r = x * w + b. 
  - En el caso de ser una neurona y dos características de entrada: r = x1 * w1 + x2 * w2 + b.
- Las **funciones de activación** pretenden introducir no linealidad en la red, las más usadas son ReLu = max(r, 0) y softmax, usada para resolver problemas de clasificación multiclase. Se hace una predicción, que es el resultado al aplicar la función de activación: a = ŷ = f(r).

2. **Propagación hacia atrás**: de salida a entrada, cuyo objetivo es actualizar los pesos y términos independientes.
- La función de pérdida evalúa el error al comparar la salida predicha con la salida real: L(y, ŷ). Existen múltiples métodos para calcularla, por ejemplo, está *binary cross-entropy* para clasificación binaria y *sparse categorical cross-entropy* para clasificación multiclase.
- Se emplea un algoritmo de optimización respecto a la función de pérdida para actualizar los pesos y términos independientes. Uno de estos métodos es el descenso por gradiente: w = w - α * ∇L(y, ŷ). 

En la fase de actualización, se emplea un parámetro llamado **tasa de aprendizaje (α)** para controlar la magnitud de los ajustes realizados en los pesos de la red neuronal durante cada paso de entrenamiento. Una tasa de aprendizaje muy grande puede provocar oscilaciones y dificultar la convergencia al punto óptimo, mientras que una tasa muy pequeña puede prolongar significativamente el tiempo de entrenamiento y el consumo de recursos computacionales

El entrenamiento se detiene cuando:
- Se ha alcanzado el número máximo de épocas indicado por el usuario.
- Se ha alcanzado la precisión deseada.
- El error de validación diverge del error de entrenamiento, lo cual significa que estamos sobreajustando la red.
<figure class="align-center" style="max-width: 90%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/error.png" alt="">
</figure>

### Redes neuronales convolucionales
---

Las redes neuronales convolucionales o **CNN** se usan para la clasificación de imágenes. Estas imágenes pueden tener dos dimensiones (filas x columnas), lo que corresponde a imágenes en escala de grises, o tres dimensiones (filas x columnas x color), correspondientes a imágenes en RGB.

En el siguiente ejemplo, podemos observar las diferentes capas que componen una CNN diseñada para un conjunto de datos en escala de grises (2D). Analizaremos cada una de estas capas:
```python
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```
<figure class="align-center" style="max-width: 80%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/CNN.png" alt="">
</figure>

#### Capa de convolución
Se aplican *kernels* (o filtros) de dimensiones nxn  para extraer características locales de la imagen. , el *kernel* se va deslizando a los largo de la imagen. El *kernel* se va deslizando a lo largo de la imagen, calculando la suma ponderada de los píxeles en cada ubicación. Cada filtro produce un mapa de características que contiene las características relevantes de la imagen. En el ejemplo proporcionado, se aplican 32 filtros en la primera y cuarta capa de convolución y 64 en la segunda y tercera.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/conv.jpeg" alt="">
</figure>

La operación de convolución reduce las dimensiones de la matriz de características. Para mantener las dimensiones constante o evitar que lleguen a cero, podemos aplicar la técnica de ***padding***, que consiste en aumentar las dimensiones añadiendo ceros sin modificar la información original. El *padding* es un parámetro flexible que puede añadirse a lo largo de toda la imagen, solo en la parte superior o en cualquier combinación deseada.
<figure class="align-center" style="max-width: 90%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/padding.png" alt="">
</figure>

Otro parámetro importante es el ***stride***, que determina el número de píxeles que el *kernel* se desplaza dentro de la imagen. Este desplazamiento se aplica tanto en filas como en columnas. En el ejemplo anterior, el *stride* es uno, ahora consideremos un caso donde sea igual a dos:
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/stride.jpeg" alt="">
</figure>

#### Capa Pooling
Estas capas reducen las dimensiones del mapa de características preservando la información más importante. Al igual que en la convolución, se desliza un *kernel* sobre la imagen. Aunque el método más común es el ***max pooling***, también existe el *average pooling*.
<figure class="align-center" style="max-width: 90%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/pooling.jpeg" alt="">
</figure>

#### Capa Flatten
Convierte los datos de entrada tridimensionales a un vector unidimensional.

#### Capa fully connected
Se corresponde con la capa *dense* del ejemplo.

#### Capa de salida o clasificación
Como ya mencionamos anteriormente, el número de neuronas es igual al número de posibles clases de salida. Usamos la función de activación *softmax*, la cual calcula la probabilidad de que un dato pertenezca a cada una de las posibles clases.

### Redes neuronales recurrentes
---

Las redes neuronales recurrentes o **RNN** buscan solucionar problemas en los que existen dependencias temporales entre características; las redes neuronales convencionales no son capaces de resolverlos de forma eficiente. La aplicación principal es el procesamiento del lenguaje natural, lo cual nos sirve para hacer traducciones, interpretar discursos o generar texto. Un ejemplo real es reconocer emociones en reseñas sobre películas, analizando si los adjetivos sean positivos o negativos.

Necesitamos transformar una frase de un máximo de *p* palabras en una entrada compatible para una red neuronal. Para lograrlo, necesitamos un diccionario que traduzca el texto a *tokens* según su índice. Este proceso se conoce como ***language processing problem***.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/language.jpg" alt="">
</figure>

Las RNN incorporan marcas de tiempo, ***timestamps***, para abordar la importancia del orden en la secuencia de datos. Por ejemplo, para los humanos la frase '*I love cats*' es comprensible, mientras que '*I cats love*' no lo es, lo que ilustra la relevancia del orden en el lenguaje natural.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/structure.jpg" alt="">
</figure>

1. Propagación hacia delante:
- a<0> = vec(0)
- a\<t> = f(waa * a\<t-1> + wax * x\<t> + b)
- y\<t> = f(way * a\<t> + b)
2. Propagación hacia atrás:
- L(y, ŷ) = ∑L\<t>(y\<t>, ŷ\<t>)
- waa = waa - αaa * ∇aaL(y, ŷ)
- way = way - αay * ∇ayL(y, ŷ)
- wax = wax - αax * ∇axL(y, ŷ)

Existen diversas estructuras de RNN que podemos seleccionar según el tipo de dataset:
- **GRU** (*Gated Recurrent Unit*): recomendada para casos donde se requiere más memoria. Por ejemplo, en la frase "*My dad, who works a lot of hours in a factory and ..., was hungry.*", la red debe ser capaz de reconocer que "*was*" se refiere al sustantivo "*dad*", mencionado bastantes palabras antes.
- ***Bi-Directional RNN***: son útiles en casos donde el contexto es relevante. Por ejemplo: "*Tim is high on drags*" / "*Tim is high in the sky*"; en el primer caso, Tim se refiere a una persona, mientras que en el segundo, se refiere a un pájaro. Es necesario reescribir la fórmula de combinación lineal: y\<t> = f(way * [af\<t>, ab\<t>] + b), donde *af* representa la propagación desde *a0* hasta *aT*, y *ab* representa la propagación desde *aT* hasta *a0*.
- **LSTM** (*Long Short-Term Memory*): adecuada para procesar frases muy extensas e incluso párrafos. Se añade una nueva salida c a la estructura convencional de las RNNs.
<figure class="align-center" style="max-width: 95%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/perception/LSTM.jpg" alt="">
</figure>SA-1B

## SAM 

intro de los 3 componentes y  foto, luego ir componente a componente 

Los modelos fine-tuned, o modelos ajustados, se refieren a modelos de inteligencia artificial que han sido entrenados en una tarea específica o en un conjunto de datos específico después de haber sido pre-entrenados en un conjunto de datos más amplio. Este proceso de ajuste fino implica tomar un modelo pre-entrenado, que ha aprendido representaciones generales de datos de un conjunto de datos grande y diverso, y luego ajustar los pesos del modelo utilizando datos más específicos o tareas adicionales.

Un proyecto de SA, **Segmentation Anything**, está compuesto por: tarea o *task*, SAM y datos (dataset de entrada y *data engine*).
- *Segmention Anything Model* o SAM: flexible promptind and provide realtime outputs para proporciona run comportamiento interactivo, ambiguity-aware -> multym mask for a single promt
- Es un modelo implementado con *prompt engineering*: el usuario da indicaiones para guiar al modelo -> que segmentar en la iamgen
- zero-shot / few-shot: modelo puede realizar la tarea sin haber sido explícitamente entrenado para ella, simplemente siguiendo las indicaciones dadas en el prompt y sin entrenamiento adicional con nueva simagnes. -----> resultados impresionantes
- Los encoder que s eutilizan para el promt o dos datos son CNN o RNN, dependiendo si son lenguaje escrito o imagenes: se quedan con las cracterística srelavantes, esas pasana al entrda del decofificador permite la sintesis d einfo entre pormt e imagen - resaltar la zona pedida creando la mascara de segmentacion ---> SAM
- el modleo s eentrana con tareas que ayuden a una buena generalizacion
- datset: Our final dataset, SA-1B, includes more than 1B masks from 11M licensed and privacy-preserving image -> responsabilida ia -> variaedad de paises y personas para que se adepte de forma igualitaria a tod oene le mundo real
- data engine -> se necesitan muechos y variados para una buena generalizaciön -> resolver el problema de que las mascras/filtros no son abundantes:?????

#### Task


## EfficientVit

## Aplicación

Red de segmentación semántica.
