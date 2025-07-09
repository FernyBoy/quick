# Entropic Associative Memory using a single register

This repository contains the procedures to replicate the experiments presented in the paper

Pineda, Luis A. & Rafael Morales (under review). _Imagery in the Entropic Associative Memory_.

The code was written in Python 3, using the Anaconda Distribution, and was run on a laptop with the following specifications:
* Model: Alienware M18 R1
* CPU: AMD Ryzen 9 7845HX
* GPU: NVIDIA GeForce RTX 4070 Max-Q (Lovelace, 8 lanes)
* RAM: 32GiB
* OS: Arch Linux (kernel 6.14.2-arch1-1)

---

## Descripción del Proyecto

Este proyecto implementa y evalúa un sistema híbrido de inteligencia artificial que combina **redes neuronales convolucionales (CNNs)** con un modelo de **Memoria Asociativa Entrópica (EAM)**. El objetivo es investigar las propiedades de la memoria, su capacidad para almacenar y recuperar patrones (imágenes), y explorar fenómenos emergentes como la "ensoñación" (dreaming).

El sistema funciona en varias etapas clave:

1.  **Percepción y Representación:** Una red neuronal (específicamente, un **autoencoder**) aprende a "percibir" las imágenes del conjunto de datos QuickDraw. El **codificador** de esta red transforma cada imagen en un vector de características de baja dimensión, que es una representación compacta y significativa de la imagen original.
2.  **Almacenamiento en Memoria:** Estos vectores de características se almacenan en la Memoria Asociativa Entrópica. La EAM no es una memoria digital convencional, sino un modelo probabilístico que almacena patrones reforzando asociaciones en una matriz interna.
3.  **Evaluación Rigurosa:** El sistema realiza una búsqueda exhaustiva de los mejores hiperparámetros para la EAM (como su tamaño y nivel de llenado) para optimizar su rendimiento en tareas de recuperación de información.
4.  **Recuperación y Visualización:** Una vez optimizada, la memoria se utiliza para recuperar patrones a partir de pistas (imágenes de prueba, incluso con ruido). Los vectores de características recuperados se pasan al **decodificador** de la red neuronal para reconstruir las imágenes, permitiendo una evaluación visual de la calidad del "recuerdo".
5.  **Exploración de Fenómenos Emergentes:** El proyecto investiga el proceso de "ensoñación", donde el sistema entra en un bucle recurrente de recordar-decodificar-codificar, generando secuencias de imágenes que exploran el espacio de conocimiento aprendido por la memoria.

En resumen, este repositorio no solo contiene el código para un modelo de IA, sino un completo banco de pruebas experimental para analizar la sinergia entre el aprendizaje de representaciones (redes neuronales) y los sistemas de memoria asociativa de inspiración biológica.

---

## Descripción de Módulos Principales

A continuación se detallan los componentes que forman el pipeline principal del proyecto.

### `eam.py`
# Análisis del Módulo `eam.py`

## Resumen General

El archivo `eam.py` es el **orquestador principal** de todo el proyecto. Actúa como un controlador que, a través de argumentos de línea de comandos, ejecuta las diferentes etapas de los experimentos con la Memoria Asociativa Entrópica (EAM).

Este script integra la funcionalidad de todos los demás módulos (`neural_net.py`, `associative.py`, `dataset.py`, `constants.py`) para llevar a cabo un flujo de trabajo completo, desde el preprocesamiento de datos y el entrenamiento de modelos hasta la evaluación exhaustiva de la memoria y la visualización de los resultados.

---

## Flujo de Trabajo y Funcionalidad por Comandos

El script se controla mediante una interfaz de línea de comandos (CLI) definida con `docopt`. Cada opción (`-n`, `-f`, `-e`, etc.) desencadena una fase específica del experimento.

### 1. `-n`: Entrenamiento de Redes Neuronales
- **Función:** `create_and_train_network()`
- **Acción:** Llama a `neural_net.train_network()` para entrenar el sistema de redes neuronales, que incluye el autoencoder (codificador + decodificador) y el clasificador. Guarda el historial de entrenamiento y la matriz de confusión resultante.

### 2. `-f`: Generación de Características
- **Función:** `produce_features_from_data()`
- **Acción:** Utiliza el **codificador** (ya entrenado en el paso `-n`) para procesar todos los conjuntos de datos (entrenamiento, llenado, prueba y con ruido). Convierte las imágenes en vectores de características (embeddings) y los guarda en archivos `.npy`. Estos vectores son la representación que se usará para alimentar la memoria asociativa.

### 3. `-e 1`: Evaluación y Búsqueda de Parámetros
- **Función:** `run_evaluation()`
- **Acción:** Esta es una de las fases más críticas. Realiza una búsqueda sistemática para encontrar los mejores hiperparámetros para la memoria asociativa.
    - **`test_memory_sizes`**: Itera sobre una lista de posibles tamaños de memoria (`constants.memory_sizes`). Para cada tamaño, entrena la memoria y mide su rendimiento (precisión, recall, entropía) usando validación cruzada.
    - **`test_memory_fills`**: Para los tamaños de memoria que resultaron ser óptimos, analiza cómo varía el rendimiento a medida que la memoria se llena con diferentes porcentajes del corpus de datos (ej. 10%, 20%, ..., 100%).
    - **`save_learned_params`**: Al final, guarda los mejores parámetros encontrados (los tamaños de memoria y porcentajes de llenado más eficientes) en un archivo para su uso en las siguientes fases.

### 4. `-r`: Generación de Recuerdos (Visualización)
- **Función:** `generate_memories()`
- **Acción:** Carga los mejores parámetros de memoria encontrados en la fase de evaluación (`-e`).
    1.  Crea una memoria con esa configuración óptima y la llena con los datos de entrenamiento.
    2.  Le presenta a la memoria los datos de prueba (tanto los originales como versiones con ruido) y le pide que "recuerde" los patrones (`remember()`).
    3.  Los patrones recordados (vectores de características) se pasan a través del **decodificador** (`decode_memories()`) para reconstruir las imágenes.
    4.  Guarda estas imágenes reconstruidas, permitiendo visualizar qué ha recuperado la memoria y comparar la calidad del recuerdo para datos limpios y con ruido.

### 5. `-d`: Proceso de "Ensoñación" (Dreaming)
- **Función:** `dream()`
- **Acción:** Implementa un fascinante ciclo recurrente para explorar el espacio latente aprendido por el sistema.
    1.  Se parte de un patrón inicial (una imagen de prueba).
    2.  La memoria asociativa "recuerda" un patrón a partir de éste (el "sueño").
    3.  El "sueño" (un vector de características) se decodifica para generar una imagen.
    4.  Esta nueva imagen se vuelve a codificar para obtener un nuevo patrón.
    5.  Este nuevo patrón se convierte en la entrada para el siguiente ciclo de "ensoñación".
    - Este proceso permite ver cómo el sistema transita por su "espacio de conocimiento", generando secuencias de imágenes que evolucionan a partir de un estímulo inicial.

---

## Funciones de Soporte y Visualización

- **Funciones de Ploteo (`plot_*`):** Un conjunto de funciones dedicadas a crear visualizaciones de los resultados usando `matplotlib` y `seaborn`. Generan gráficas de precisión vs. recall, matrices de confusión, comportamiento de la memoria, etc.
- **Funciones de Cuantización (`msize_features`, `rsize_recall`):** Convierten los vectores de características (que son de punto flotante) a enteros para que puedan ser usados por la memoria, y viceversa.
- **Internacionalización:** Utiliza `gettext` para que los textos en las gráficas puedan mostrarse en español o inglés, lo que demuestra un alto nivel de detalle en la implementación.

## Conclusión

`eam.py` es el cerebro del proyecto. No solo ejecuta una secuencia de pasos, sino que implementa un completo banco de pruebas científico. Orquesta la interacción entre el aprendizaje perceptivo (redes neuronales) y el almacenamiento/recuperación de patrones (memoria asociativa), y proporciona las herramientas para evaluar rigurosamente el sistema y visualizar sus resultados de manera significativa.

### `associative.py`
# Análisis del Módulo `associative.py`

## Resumen General

Este archivo define la implementación de un modelo de **Memoria Asociativa**. No se trata de una memoria de computadora convencional, sino de un modelo computacional inspirado en cómo podría funcionar la memoria biológica. Su propósito es almacenar patrones (vectores de datos) y recuperarlos más tarde, incluso a partir de pistas incompletas o con ruido.

El archivo contiene dos clases principales:
1.  `AssociativeMemory`: Es la implementación de una única memoria asociativa.
2.  `AssociativeMemorySystem`: Es un gestor que maneja un conjunto de múltiples memorias `AssociativeMemory`, generalmente una para cada "clase" o "categoría" de datos que se quiere aprender.

---

## Clase `AssociativeMemory`

Esta es la clase central. Representa una única memoria capaz de aprender y recordar patrones.

### 1. Estructura de la Memoria

- **`_relation` (Matriz de Relación):** El corazón de la memoria. Es una matriz 2D de NumPy (`m x n`), donde `n` es el número de características del patrón (el "dominio") y `m` es el número de posibles valores que cada característica puede tomar (el "rango").
- El valor en `_relation[i, j]` representa la **fuerza de la asociación** entre la característica `j` y su valor `i`. Cada vez que se registra un patrón, los valores correspondientes en esta matriz se incrementan.

### 2. Parámetros Clave (Hiperparámetros)

La memoria se configura con varios parámetros que definen su comportamiento:

- **`xi` (tolerancia `_t`):** El número máximo de características que pueden no coincidir en un patrón para que aún sea reconocido. Es un umbral de error simple.
- **`sigma`:** La desviación estándar de una función gaussiana (`normpdf`). Se usa durante la recuperación para crear una "búsqueda difusa". En lugar de buscar una coincidencia exacta, los valores cercanos al valor de la pista también se consideran, con una fuerza que disminuye con la distancia. Un `sigma` más grande implica una memoria más "borrosa" o flexible.
- **`iota`:** Un umbral para filtrar el ruido. Al reconocer un patrón, la memoria ignora las asociaciones en la matriz `_relation` que son más débiles que un promedio local ponderado por `iota`. Esto evita que las asociaciones débiles o espurias afecten el reconocimiento.
- **`kappa`:** Un umbral de confianza. Para que un patrón sea reconocido, el "peso" promedio de sus características (la fuerza de sus asociaciones en la memoria) debe superar un umbral global definido por `kappa`. Esto asegura que solo los patrones que coinciden con asociaciones fuertes sean recuperados.

### 3. Operaciones Principales

- **`register(vector)`:** La operación de "escritura" o "aprendizaje".
    1.  Toma un vector de entrada.
    2.  Incrementa los contadores en la matriz `_relation` en las posiciones correspondientes a cada par `(valor, característica)` del vector.
    3.  Esto "graba" el patrón en la memoria, reforzando sus asociaciones.

- **`recall(vector)`:** La operación de "lectura" o "recuerdo". Es el proceso más complejo.
    1.  **Validación (`recognize`):** Primero, determina si el vector de entrada es "suficientemente similar" a algo que la memoria conoce. Esto implica dos comprobaciones:
        -   El número de características que no coinciden debe ser menor o igual a la tolerancia (`xi`).
        -   La fuerza promedio de las asociaciones del patrón debe superar el umbral de confianza (`kappa`).
    2.  **Recuperación (`lreduce`):** Si el patrón se reconoce, la memoria construye un nuevo patrón de salida. Para cada característica, en lugar de devolver un valor fijo, lo **elige probabilísticamente** de la columna correspondiente en la matriz `_relation`, usando la distribución gaussiana (definida por `sigma`) centrada en el valor de la pista. Esto permite a la memoria "rellenar" partes faltantes o corregir errores en la pista.
    3.  Si el patrón no se reconoce, devuelve un vector que indica "indefinido".

- **`entropy` (Entropía):** La memoria puede calcular su propia entropía. En este contexto, la entropía es una medida de la **incertidumbre** o **ambigüedad** en la memoria. Una columna con alta entropía significa que la memoria ha aprendido muchas asociaciones diferentes para esa característica, lo que la hace menos "segura" de cuál es el valor correcto.

---

## Clase `AssociativeMemorySystem`

Esta clase actúa como un director de orquesta para múltiples memorias.

- **Inicialización:** Se crea con una lista de etiquetas (por ejemplo, las clases de un problema de clasificación: '0', '1', '2', ...). Crea una instancia de `AssociativeMemory` para cada etiqueta.
- **`register(mem, vector)`:** Registra un vector en la memoria específica asociada a la etiqueta `mem`. Así, todos los ejemplos de la clase '0' van a la memoria del '0', y así sucesivamente.
- **`recall(vector)`:**
    1.  Presenta el vector de entrada a **todas** las memorias que gestiona.
    2.  Cada memoria intenta recordar un patrón y devuelve si tuvo éxito y con qué "calidad" (una combinación de su entropía y el peso del recuerdo).
    3.  El sistema elige el recuerdo de la memoria que respondió con la "mejor" calidad (menor penalización). La etiqueta de esa memoria se convierte en la clasificación del sistema para el vector de entrada.

### Conclusión

El archivo `associative.py` proporciona un sistema de memoria sofisticado que va más allá de la simple coincidencia de patrones. Utiliza principios estadísticos y probabilísticos para lograr un comportamiento flexible y robusto:

-   **Almacena** patrones reforzando asociaciones en una matriz.
-   **Recuerda** patrones de forma difusa, permitiendo la recuperación a partir de pistas incompletas o con ruido.
-   Utiliza un conjunto de **hiperparámetros** (`xi`, `sigma`, `iota`, `kappa`) para ajustar finamente su comportamiento entre la precisión estricta y la flexibilidad.
-   El `AssociativeMemorySystem` lo convierte en un **sistema de clasificación**, donde la "mejor" memoria en responder determina la clase del patrón de entrada.

### `neural_net.py`
# Análisis del Módulo `neural_net.py`

## Resumen General

Este archivo es el responsable de todo lo relacionado con las **redes neuronales** del proyecto. Su función principal es definir, entrenar y utilizar varios modelos de redes neuronales construidos con `TensorFlow/Keras`. Estos modelos son cruciales para transformar los datos crudos (probablemente imágenes) en una representación más compacta y significativa (los "vectores de características") que luego será utilizada por la memoria asociativa.

El archivo define tres arquitecturas de red principales que se combinan para formar un sistema completo:

1.  **Encoder (Codificador):** Comprime los datos de entrada.
2.  **Decoder (Decodificador):** Reconstruye los datos originales a partir de la versión comprimida.
3.  **Classifier (Clasificador):** Predice la clase de los datos a partir de la versión comprimida.

---

## Arquitecturas de los Modelos

### 1. `get_encoder()`

- **Propósito:** Reducir la dimensionalidad de los datos de entrada. Transforma una imagen (ej. 28x28 píxeles) en un vector plano de tamaño fijo (`constants.domain`).
- **Arquitectura:** Es una **Red Neuronal Convolucional (CNN)**.
    - Utiliza una serie de bloques `conv_block`, cada uno compuesto por:
        - Capas `Conv2D` para detectar características locales en las imágenes (bordes, texturas, etc.).
        - `BatchNormalization` para estabilizar y acelerar el entrenamiento.
        - `MaxPool2D` para reducir el tamaño de los mapas de características y hacer la representación más robusta a pequeñas traslaciones.
        - `SpatialDropout2D` para regularizar el modelo y prevenir el sobreajuste.
    - La salida de los bloques convolucionales se aplana (`Flatten`) y se normaliza (`LayerNormalization`) para producir el vector de características final, también llamado "código" o "embedding".

### 2. `get_decoder()`

- **Propósito:** Realizar la operación inversa al codificador. Toma un vector de características y trata de reconstruir la imagen original.
- **Arquitectura:** Es una **Red Neuronal Convolucional Transpuesta (a veces llamada Deconvolucional)**.
    - Comienza con una capa `Dense` para expandir el vector de características.
    - Usa `Reshape` para convertir el vector expandido en un tensor con forma de imagen pequeña.
    - Aplica capas `Conv2D` y `UpSampling2D` para aumentar gradualmente el tamaño de la imagen y refinar los detalles, revirtiendo el proceso del `MaxPool2D` del codificador.
    - La capa final es una `Conv2D` con activación `sigmoid`, que produce una imagen normalizada (valores de píxeles entre 0 y 1).

### 3. `get_classifier()`

- **Propósito:** Tomar un vector de características (la salida del codificador) y predecir a qué clase pertenece.
- **Arquitectura:** Es una **Red Neuronal Densa (o Perceptrón Multicapa)** simple.
    - Consiste en varias capas `Dense` con activación `relu`.
    - Usa `Dropout` entre las capas para regularización.
    - La capa final es una `Dense` con activación `softmax`, que produce una distribución de probabilidad sobre todas las clases posibles. La clase con la mayor probabilidad es la predicción del modelo.

---

## Procesos Clave

### 1. `train_network(prefix, es)`

- **Orquestación del Entrenamiento:** Esta es la función principal que une todos los componentes.
    - **Modelos Combinados:** No entrena los modelos por separado. Crea dos modelos compuestos:
        1.  **Autoencoder:** `encoder` -> `decoder`. Se entrena para minimizar la diferencia entre la imagen original y la reconstruida (pérdida `mean_squared_error` o `huber`). Su objetivo es aprender una buena compresión y descompresión de los datos.
        2.  **Full Classifier:** `encoder` -> `classifier`. Se entrena para clasificar correctamente las imágenes (pérdida `categorical_crossentropy`).
    - **Entrenamiento Multi-Tarea:** Lo más importante es que crea un **modelo final combinado** que toma una imagen como entrada y produce **dos salidas**: la clasificación (del clasificador) y la imagen reconstruida (del decodificador). Este modelo se entrena simultáneamente en ambas tareas. Esto es una técnica poderosa, ya que obliga al codificador a aprender representaciones (features) que no solo son buenas para reconstruir la imagen, sino también para clasificarla.
    - **Validación Cruzada:** Itera a través de `constants.n_folds` para realizar validación cruzada, asegurando que el rendimiento del modelo sea robusto y no dependa de una partición específica de los datos.
    - **Early Stopping:** Utiliza una clase `Callback` personalizada (`EarlyStopping`) para detener el entrenamiento si el rendimiento en el conjunto de validación deja de mejorar, evitando el sobreajuste y ahorrando tiempo computacional.
    - **Guardado de Modelos:** Al final de cada pliegue (fold) de la validación cruzada, guarda los modelos entrenados (`encoder`, `decoder`, `classifier`) en archivos para su uso posterior.

### 2. `obtain_features(...)`

- **Inferencia:** Una vez que los modelos están entrenados, esta función se utiliza para procesar todos los conjuntos de datos (entrenamiento, prueba, etc.).
    - Carga el modelo `encoder` previamente guardado.
    - Pasa los datos a través del codificador para obtener los vectores de características.
    - Guarda estos vectores de características en archivos `.npy`. Estos archivos son la entrada directa para el sistema de memoria asociativa (`eam.py`).

---

## Conclusión

`neural_net.py` es el motor de "percepción" del sistema. Su rol es tomar datos crudos y complejos (imágenes) y destilar su esencia en vectores de características compactos y ricos en información. La arquitectura de entrenamiento multi-tarea es particularmente sofisticada, ya que asegura que estas características sean útiles tanto para la clasificación como para la reconstrucción, lo que probablemente conduce a una representación más robusta y generalizable. Este módulo prepara los datos de una manera que la memoria asociativa pueda procesarlos de manera efectiva.

### `dataset.py`
# Análisis del Módulo `dataset.py`

## Resumen General

Este archivo es el **módulo de gestión de datos** del proyecto. Su única responsabilidad es cargar, preprocesar, dividir y servir los datos de entrenamiento y prueba a los otros componentes del sistema, como el entrenador de redes neuronales (`neural_net.py`) y el evaluador de la memoria asociativa (`eam.py`).

El módulo está diseñado para ser eficiente y robusto, incorporando mecanismos de **caching (almacenamiento en caché)**, **preprocesamiento** (normalización, adición de ruido, barajado) y una lógica de **división de datos para validación cruzada**.

---

## Funcionalidad Principal

### 1. Carga de Datos y Caching

- La función principal `_load_dataset` orquesta la carga de datos.
- Primero, intenta cargar un conjunto de datos ya preprocesado y guardado en archivos `.npy` (`prep_data.fname`, etc.). Esto se hace en la función `_preprocessed_dataset`.
- Si estos archivos no existen (por ejemplo, en la primera ejecución), procede a cargar los datos crudos desde su fuente original.
- Una vez que los datos crudos se cargan y procesan, se guardan en el disco (`_save_dataset`) para que las ejecuciones posteriores sean mucho más rápidas.

### 2. Fuente de Datos: Google QuickDraw

- La función `_load_quickdraw` revela la fuente de datos: el **conjunto de datos QuickDraw de Google**.
- Carga los datos desde archivos `.npy` (por ejemplo, `full_numpy_bitmap_cat.npy`, `full_numpy_bitmap_dog.npy`).
- **Limitación y Balanceo de Clases:**
    - Limita el número de clases a cargar según el valor de `constants.n_labels`.
    - Para asegurar que el conjunto de datos esté balanceado (misma cantidad de ejemplos por clase), detecta la clase con el menor número de imágenes y recorta todas las demás clases a ese mismo tamaño.

*(Nota: Existe una función `_load_mnist` que no parece ser utilizada, lo que sugiere que el código fue adaptado o podría ser extendido para usar el dataset MNIST, pero actualmente está enfocado en QuickDraw).*

### 3. Preprocesamiento de Datos

Una vez cargados los datos crudos, se aplican varios pasos de preprocesamiento:

- **Normalización:** Los valores de los píxeles de las imágenes (originalmente de 0 a 255) se escalan a un rango de 0.0 a 1.0.
- **Inyección de Ruido (`noised`):** Se crea una copia completa del conjunto de datos a la que se le añade ruido. La función `_noised` toma una imagen y reemplaza un porcentaje de sus píxeles (`constants.noise_percent`) con valores aleatorios. Esto es fundamental para evaluar la robustez de los modelos y su capacidad para reconocer patrones dañados.
- **Barajado (`_shuffle`):** Los datos y sus etiquetas se barajan de forma aleatoria. Esto es crucial para asegurar que los modelos de aprendizaje no aprendan ningún sesgo relacionado con el orden original de los datos.

### 4. División de Datos para Validación Cruzada

- Esta es la funcionalidad más importante del módulo. Las funciones públicas (`get_training`, `get_filling`, `get_testing`) permiten a otros scripts solicitar segmentos específicos del conjunto de datos.
- La función interna `_get_segment` implementa una estrategia de **validación cruzada (k-fold cross-validation)**.
    - El conjunto de datos completo se trata como un **búfer circular**.
    - Para cada "pliegue" (`fold`) de la validación, la ventana de datos de entrenamiento, llenado y prueba se desplaza a lo largo de este búfer.
    - Los tamaños de estas ventanas se definen como porcentajes en `constants.py` (`nn_training_percent`, `am_filling_percent`, `am_testing_percent`).
- Esto asegura que los modelos se entrenen y evalúen en diferentes subconjuntos de datos en cada pliegue, lo que proporciona una medida mucho más fiable de su rendimiento general.

### 5. Formato de Salida

- Los datos de las imágenes se devuelven como arrays de NumPy.
- Las etiquetas se convierten a formato **one-hot encoding** usando `keras.utils.to_categorical`. Por ejemplo, la etiqueta `3` en un problema de 4 clases se convierte en el vector `[0, 0, 0, 1]`. Este formato es el que requieren las funciones de pérdida como `categorical_crossentropy` en Keras.

---

### Conclusión

`dataset.py` es un módulo de pipeline de datos bien estructurado y autocontenido. Abstrae toda la complejidad de la carga, el procesamiento y la división de los datos. Sus características clave son:

- **Eficiencia:** Gracias al caching de datos preprocesados.
- **Robustez:** Al implementar una estrategia de validación cruzada que garantiza una evaluación fiable del modelo.
- **Flexibilidad:** Al generar conjuntos de datos con y sin ruido para pruebas de robustez.

En esencia, proporciona una API simple y consistente para que el resto del código obtenga los datos exactos que necesita para cualquier fase del experimento, sin tener que preocuparse por los detalles de la implementación.

### `constants.py`
# Análisis del Módulo `constants.py`

## Resumen General

Este archivo es el **centro de configuración** de todo el proyecto. Su propósito es centralizar en un único lugar todos los parámetros, constantes, nombres de archivo, rutas y valores por defecto que se utilizan en los demás scripts (`eam.py`, `neural_net.py`, `associative.py`, etc.).

Centralizar la configuración de esta manera es una práctica de programación excelente por varias razones:
1.  **Facilita la Modificación:** Si se necesita cambiar un parámetro (como el número de épocas de entrenamiento o el tamaño de la memoria), solo hay que hacerlo en este archivo, en lugar de buscarlo en múltiples scripts.
2.  **Consistencia:** Asegura que todos los módulos del proyecto usen los mismos valores y, lo que es más importante, las mismas convenciones para nombrar archivos y directorios. Esto es crucial para que los datos generados por un script puedan ser leídos correctamente por otro.
3.  **Legibilidad:** Mantiene los otros archivos más limpios y enfocados en su lógica, en lugar de estar llenos de valores "mágicos" (números o cadenas de texto sin un significado claro).

---

## Secciones Principales del Archivo

### 1. Rutas y Directorios

- Define las rutas base donde se guardarán todos los resultados (`run_path`) y los datos procesados (`data_path`).
- Especifica subdirectorios para diferentes tipos de salidas, como imágenes (`image_path`), recuerdos de la memoria (`memories_path`) y "sueños" (`dreams_path`).

```python
# Directory where all results are stored.
data_path = 'data/quick'
run_path = 'runs'
image_path = 'images'
testing_path = 'test'
memories_path = 'memories'
dreams_path = 'dreams'
```

### 2. Prefijos y Sufijos para Nombres de Archivo

- Esta es una de las partes más importantes. Define un sistema estandarizado de prefijos y sufijos para construir los nombres de los archivos de manera programática.
- Por ejemplo, un archivo que contiene las "características" (`features_prefix`) del conjunto de "entrenamiento" (`training_suffix`) para el "pliegue 0" (`fold_suffix(0)`) se construirá de manera consistente.
- Esto permite que el código genere y encuentre archivos de forma fiable sin tener que codificar nombres de archivo completos en todas partes.

```python
data_prefix = 'data'
features_prefix = 'features'
training_suffix = '-training'
testing_suffix = '-testing'
encoder_suffix = '-encoder'
```

### 3. Parámetros del Experimento y del Modelo

- **Parámetros de la Red Neuronal:** Define constantes para el entrenamiento, como el número de pliegues para la validación cruzada (`n_folds`), el número de ciclos en el proceso de "ensoñación" (`dreaming_cycles`), etc.
- **Parámetros de la Memoria Asociativa:**
    - Define los valores por defecto para los hiperparámetros de la memoria: `iota_default`, `kappa_default`, `xi_default`, `sigma_default`.
    - Define los rangos de valores que se probarán en los experimentos de evaluación, como los diferentes tamaños de memoria (`memory_sizes`) y los porcentajes de llenado (`memory_fills`).
- **Parámetros de los Datos:** Define el número de clases (`n_labels`), el porcentaje de datos para entrenamiento vs. prueba, etc.

```python
# Number of columns in memory
domain = 256
n_folds = 1
dreaming_cycles = 6

# Memory parameters
iota_default = 0.0
sigma_default = 0.25

# Experiment ranges
memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
```

### 4. Índices y Etiquetas

- Define constantes para acceder a los resultados de manera legible. Por ejemplo, en lugar de usar `resultados[4]` para obtener la métrica "sin respuesta", se puede usar `resultados[no_response_idx]`, lo que hace el código mucho más claro.

```python
precision_idx = 0
recall_idx = 1
no_response_idx = 4
n_behaviours = 7
```

### 5. Clase `ExperimentSettings`

- Es una clase simple que encapsula los parámetros de un experimento específico. Esto permite pasar toda la configuración de un experimento como un único objeto, en lugar de pasar múltiples parámetros individuales a las funciones.

### 6. Funciones de Ayuda (Helpers)

- **Funciones para Nombres de Archivo:** Contiene un conjunto de funciones (`filename`, `csv_filename`, `data_filename`, `encoder_filename`, etc.) que utilizan los prefijos y sufijos definidos anteriormente para construir nombres de archivo completos y estandarizados. Estas funciones son el pegamento que une el sistema de nomenclatura.
- **Funciones de Ayuda para el Sistema de Archivos:** Incluye `create_directory` para asegurarse de que los directorios de salida existan antes de intentar escribir en ellos.
- **Funciones de Impresión y Formato:** Pequeñas utilidades para imprimir mensajes de advertencia, contadores de progreso y formatear números en cadenas.

```python
def encoder_filename(name_prefix, es, fold):
    return filename(name_prefix + encoder_suffix, es, fold) + ".keras"

def create_directory(path):
    # ...
```

### Conclusión

`constants.py` es la columna vertebral organizativa del proyecto. No contiene lógica de negocio compleja, pero su papel es fundamental para la mantenibilidad, escalabilidad y robustez del código. Al definir un "lenguaje común" para los parámetros y los nombres de archivo, permite que los diferentes componentes del sistema (procesamiento de datos, entrenamiento de redes neuronales, experimentos con la memoria asociativa) se comuniquen e interactúen de manera fluida y predecible. Sin un archivo como este, el proyecto sería mucho más frágil y difícil de gestionar.

---
## Scripts de Utilidad y Análisis

Esta sección describe varios scripts que no forman parte del pipeline principal de ejecución, sino que se utilizan para preparar datos específicos, analizar resultados o agregar métricas de múltiples experimentos.

### `choose.py`
# Análisis del Módulo `choose.py`

### Resumen General

El script `choose.py` es un programa de utilidad diseñado para realizar una tarea de preparación muy específica: **seleccionar un único ejemplo de datos de cada pliegue (fold) de la validación cruzada para ser utilizado como "semilla" en los experimentos de "ensoñación" (dreaming)**.

Su objetivo es encontrar, para cada fold, un ejemplo que cumpla con un criterio clave: que haya sido **clasificado correctamente** por la red neuronal. Esto asegura que el proceso de ensoñación comience a partir de un punto de partida "conocido" y estable para el sistema, en lugar de un ejemplo que la red ya encuentra confuso.

### Proceso de Selección

El script sigue una lógica metódica para elegir estos ejemplos semilla:

1.  **Inicialización:** Crea una estructura para almacenar la etiqueta y el índice del ejemplo elegido para cada fold (`chosen`).
2.  **Asignación de Clases:** Baraja aleatoriamente las clases de datos para asignar una clase objetivo diferente a cada fold.
3.  **Iteración por Fold:** Recorre cada uno de los pliegues de la validación cruzada.
4.  **Carga de Datos:** Para el fold actual, carga dos archivos cruciales:
    *   Las **etiquetas verdaderas** del conjunto de prueba.
    *   Las **etiquetas predichas** por el clasificador de la red neuronal para ese mismo conjunto.
5.  **Búsqueda del Ejemplo Ideal:** Itera sobre las etiquetas verdaderas y predichas buscando un ejemplo que satisfaga tres condiciones:
    *   La etiqueta verdadera debe coincidir con la clase objetivo para ese fold.
    *   La etiqueta predicha debe coincidir con la etiqueta verdadera (es decir, el ejemplo fue **clasificado correctamente**).
    *   Se introduce un elemento de azar (`random.randrange(10) == 0`) para no elegir siempre el primer ejemplo que encuentre, sino uno al azar entre los candidatos.
6.  **Almacenamiento:** Una vez que encuentra un ejemplo que cumple los requisitos, guarda su **etiqueta** y su **índice** en la estructura `chosen` y pasa al siguiente fold.

### Salida del Script

*   El resultado final es un archivo llamado `chosen.csv`.
*   Este archivo contiene una fila por cada fold, con dos columnas: la **etiqueta de la clase** del ejemplo elegido y el **índice** de ese ejemplo dentro del conjunto de datos de prueba de ese fold.

### Conclusión

`choose.py` es un paso preparatorio indispensable para los experimentos de ensoñación (`eam.py -d`). Al seleccionar cuidadosamente un punto de partida estable y bien aprendido para cada fold, permite que los experimentos posteriores investiguen de manera más controlada cómo la memoria asociativa genera y transforma patrones a partir de un "recuerdo" inicial coherente.

### `classif.py`
# Análisis del Módulo `classif.py`

### Resumen General

`classif.py` es, al igual que `classif_dreams.py`, un script de **análisis post-experimento**. Su función no es ejecutar una parte del pipeline, sino examinar los resultados de la fase de "recuerdo" (`remember`) para entender el impacto de uno de los hiperparámetros clave de la memoria: `sigma`.

El script se centra en responder: "Para una imagen de entrada (con ruido), ¿cómo cambia su clasificación después de ser procesada por la memoria asociativa en un solo paso de recuerdo, y cómo influye el valor de `sigma` en este resultado?".

### Funcionamiento Detallado

1.  **Carga de Semillas:** Al igual que otros scripts de análisis, comienza cargando `chosen.csv` para identificar las imágenes "semilla" (correctamente clasificadas en su versión original) que sirven como casos de estudio.
2.  **Configuración Específica:** Fija un tamaño de memoria (`msize = 4`) para el análisis, lo que indica que está diseñado para estudiar el efecto de `sigma` manteniendo constante el tamaño de la memoria.
3.  **Iteración por Fold:** Recorre cada pliegue (fold) del experimento.
4.  **Análisis por Fold:**
    *   **Clasificación Directa (Línea Base):** Carga los resultados de la clasificación directa de la red neuronal sobre el conjunto de prueba **con ruido**. Imprime la etiqueta verdadera de la imagen semilla y la etiqueta que le asignó la red directamente a su versión con ruido. Este es el punto de referencia: muestra si el ruido hizo que la red se equivocara.
    *   **Iteración sobre `sigma`:** Recorre una lista predefinida de valores de `sigma`.
    *   **Clasificación Post-Recuerdo:** Para cada valor de `sigma`, carga el archivo de resultados de clasificación correspondiente a los patrones que fueron procesados (recordados) por una memoria con `msize=4` y ese `sigma` específico.
    *   Imprime la clasificación del patrón semilla *después* de haber pasado por la memoria.
5.  **Formato de Salida:** El script imprime en la consola una línea por cada fold. Cada línea es una secuencia de clasificaciones, comenzando con la clasificación directa y seguida por las clasificaciones post-recuerdo para cada valor de `sigma` probado.

### Conclusión

`classif.py` es una herramienta de análisis enfocada que permite a los investigadores **aislar y evaluar el efecto del parámetro `sigma`** en la capacidad de la memoria para "limpiar" o "corregir" un patrón con ruido. Al comparar la clasificación directa (potencialmente errónea) con las clasificaciones obtenidas tras el recuerdo con diferentes `sigma`, se puede determinar qué nivel de "flexibilidad" o "desenfoque" (`sigma`) es más beneficioso para la corrección de errores en un solo paso, una de las funciones clave de una memoria asociativa.

### `classif_dreams.py`
# Análisis del Módulo `classif_dreams.py`

### Resumen General

El script `classif_dreams.py` es un programa de utilidad diseñado para realizar una tarea de preparación muy específica: **seleccionar un único ejemplo de datos de cada pliegue (fold) de la validación cruzada para ser utilizado como "semilla" en los experimentos de "ensoñación" (dreaming)**.

Su objetivo es encontrar, para cada fold, un ejemplo que cumpla con un criterio clave: que haya sido **clasificado correctamente** por la red neuronal. Esto asegura que el proceso de ensoñación comience a partir de un punto de partida "conocido" y estable para el sistema, en lugar de un ejemplo que la red ya encuentra confuso.

### Proceso de Selección

El script sigue una lógica metódica para elegir estos ejemplos semilla:

1.  **Inicialización:** Crea una estructura para almacenar la etiqueta y el índice del ejemplo elegido para cada fold (`chosen`).
2.  **Asignación de Clases:** Baraja aleatoriamente las clases de datos para asignar una clase objetivo diferente a cada fold.
3.  **Iteración por Fold:** Recorre cada uno de los pliegues de la validación cruzada.
4.  **Carga de Datos:** Para el fold actual, carga dos archivos cruciales:
    *   Las **etiquetas verdaderas** del conjunto de prueba.
    *   Las **etiquetas predichas** por el clasificador de la red neuronal para ese mismo conjunto.
5.  **Búsqueda del Ejemplo Ideal:** Itera sobre las etiquetas verdaderas y predichas buscando un ejemplo que satisfaga tres condiciones:
    *   La etiqueta verdadera debe coincidir con la clase objetivo para ese fold.
    *   La etiqueta predicha debe coincidir con la etiqueta verdadera (es decir, el ejemplo fue **clasificado correctamente**).
    *   Se introduce un elemento de azar (`random.randrange(10) == 0`) para no elegir siempre el primer ejemplo que encuentre, sino uno al azar entre los candidatos.
6.  **Almacenamiento:** Una vez que encuentra un ejemplo que cumple los requisitos, guarda su **etiqueta** y su **índice** en la estructura `chosen` y pasa al siguiente fold.

### Salida del Script

*   El resultado final es un archivo llamado `chosen.csv`.
*   Este archivo contiene una fila por cada fold, con dos columnas: la **etiqueta de la clase** del ejemplo elegido y el **índice** de ese ejemplo dentro del conjunto de datos de prueba de ese fold.

### Conclusión

`choose.py` es un paso preparatorio indispensable para los experimentos de ensoñación (`eam.py -d`). Al seleccionar cuidadosamente un punto de partida estable y bien aprendido para cada fold, permite que los experimentos posteriores investiguen de manera más controlada cómo la memoria asociativa genera y transforma patrones a partir de un "recuerdo" inicial coherente.

### `mcols_stdevs.py`
# Análisis del Módulo `mcols_stdevs.py`

### Resumen General

`mcols_stdevs.py` es un script de **agregación y resumen de resultados** que se ejecuta después de haber completado múltiples series de experimentos. Su propósito es consolidar las métricas de rendimiento (precisión, recall, entropía) de experimentos que fueron ejecutados con diferentes tamaños para el dominio de la memoria (`constants.domain`).

El script asume que se ha ejecutado el pipeline de evaluación (`eam.py -e 1`) varias veces, y que cada ejecución, correspondiente a un tamaño de dominio diferente, ha guardado sus resultados en un directorio separado (ej. `runs-32`, `runs-64`, `runs-128`, etc.).

### Funcionamiento Detallado

1.  **Iteración sobre Tamaños de Dominio:** El script itera sobre una lista predefinida de tamaños de dominio (`domain_sizes`).
2.  **Cambio Dinámico de Directorio:** Para cada tamaño de dominio, modifica dinámicamente la variable `constants.run_path` para apuntar al directorio de resultados correspondiente (ej. `runs-128`). Esto le permite usar las funciones de `constants.py` para encontrar los archivos correctos en el lugar adecuado.
3.  **Iteración sobre Métricas:** Dentro de cada directorio, itera sobre una lista de nombres de archivos de métricas (`memory_entropy`, `memory_precision`, `memory_recall`).
4.  **Carga y Cálculo:**
    *   Carga el archivo CSV de la métrica actual. Este archivo típicamente contiene una fila por cada pliegue (fold) de la validación cruzada y una columna por cada tamaño de rango de memoria (`memory_sizes`) probado.
    *   Calcula la **media** y la **desviación estándar** a lo largo de las columnas. Esto agrega los resultados de todos los pliegues, dando una medida central del rendimiento y su variabilidad.
5.  **Impresión de Resultados:** Imprime los resultados agregados en la consola en formato CSV. Cada línea de salida contiene:
    *   El tamaño del dominio que se está analizando.
    *   El nombre de la métrica.
    *   Los valores de la media (o desviación estándar) para cada tamaño de rango de memoria.

### Conclusión

`mcols_stdevs.py` es una herramienta de análisis crucial para el proceso de investigación. Permite a los investigadores pasar de los resultados brutos de múltiples ejecuciones a una tabla resumida y fácil de interpretar. Esta tabla consolidada es fundamental para comparar el rendimiento del sistema bajo diferentes configuraciones del hiperparámetro `domain` y para sacar conclusiones sobre cuál es la configuración más óptima para la memoria asociativa.

### `nnet_stats.py`
# Análisis del Módulo `nnet_stats.py`

### Resumen General

`nnet_stats.py` es otro script de **análisis y agregación de resultados**. Su función específica es calcular la **precisión (accuracy) de la red neuronal clasificadora** en los diferentes conjuntos de experimentos.

Al igual que `mcols_stdevs.py`, este script asume que se han ejecutado múltiples pipelines de experimentación, cada uno con un tamaño de `domain` diferente, y que los resultados se encuentran en directorios separados (`runs-32`, `runs-64`, etc.).

### Funcionamiento Detallado

1.  **Función de Precisión:** Define una función de ayuda `accuracy` que calcula la precisión simple: el número de predicciones correctas dividido por el número total de ejemplos.
2.  **Iteración sobre Configuraciones:** El script itera a través de dos bucles anidados:
    *   El bucle exterior recorre los diferentes tamaños de dominio (`domain_sizes`) con los que se corrieron los experimentos.
    *   El bucle interior recorre cada pliegue (fold) de la validación cruzada para ese dominio.
3.  **Carga de Datos Relevantes:** Para cada combinación específica de `dominio` y `fold`:
    *   Modifica `constants.run_path` para apuntar al directorio de resultados correcto.
    *   Carga las **etiquetas verdaderas** del conjunto de prueba.
    *   Carga las **etiquetas predichas** por el clasificador de la red neuronal para ese mismo conjunto.
4.  **Cálculo e Impresión:**
    *   Utiliza la función `accuracy` para calcular la precisión de la red neuronal en ese fold.
    *   Imprime el resultado en la consola en un formato CSV simple: `dominio, fold, precision`.

### Conclusión

`nnet_stats.py` es una herramienta de evaluación enfocada en el componente de red neuronal del sistema. Permite a los investigadores medir el rendimiento base del clasificador antes de que la memoria asociativa entre en juego. Esta métrica es fundamental, ya que la precisión del clasificador establece un límite superior en el rendimiento que el sistema híbrido puede alcanzar. Los resultados generados por este script son esenciales para entender si los errores del sistema completo se originan en la percepción (la red neuronal) o en el proceso de memoria.

### `noised_classif.py`
# Análisis del Módulo `noised_classif.py`

### Resumen General

`noised_classif.py` es un script de **pre-cálculo y generación de datos**, no un script de análisis final. Su única función es tomar los vectores de características que fueron extraídos de las **imágenes con ruido** y pasarlos a través del clasificador de la red neuronal para ver qué predice.

El script genera un conjunto de resultados que sirve como **línea base (baseline)** para evaluar el rendimiento del sistema en condiciones no ideales.

### Funcionamiento Detallado

1.  **Iteración por Fold:** El script recorre cada uno de los pliegues (folds) de la validación cruzada.
2.  **Carga del Clasificador:** Para cada fold, carga el modelo `classifier` correspondiente que fue entrenado previamente.
3.  **Carga de Características con Ruido:** Carga los archivos de características (`features-noised_...npy`) que contienen los vectores generados por el codificador a partir de las imágenes del conjunto de prueba a las que se les añadió ruido.
4.  **Predicción:** Utiliza el clasificador cargado para predecir las etiquetas de clase para estas características "ruidosas".
5.  **Guardado de Resultados:** Guarda las etiquetas predichas en un nuevo archivo (`classif-noised_...npy`).

### Conclusión

Este script es un paso preparatorio crucial para los análisis posteriores. Crea un registro de cómo se comporta el clasificador de la red neuronal frente a datos corruptos, **antes de que la memoria asociativa tenga la oportunidad de intervenir**. El archivo de salida (`classif-noised_...npy`) representa el rendimiento "ingenuo" del sistema ante el ruido y se convierte en el punto de referencia contra el cual se compararán los resultados después de que la memoria asociativa intente "limpiar" o "corregir" estos patrones. Esto permite medir de forma cuantitativa la capacidad de la memoria para la corrección de errores.

### `parse_history.py`
# Análisis del Módulo `parse_history.py`

### Resumen General

`parse_history.py` es un script de **análisis y extracción de datos** que se ejecuta post-experimento. Su única función es leer los archivos JSON que guardan el historial de entrenamiento de las redes neuronales y extraer de ellos las métricas finales de rendimiento.

Este script permite consolidar rápidamente los resultados clave del entrenamiento de los modelos (clasificador y autoencoder) a través de diferentes configuraciones experimentales (específicamente, diferentes tamaños de `domain`).

### Funcionamiento Detallado

1.  **Iteración sobre Configuraciones:** El script itera sobre una lista predefinida de tamaños de dominio (`domain_sizes`) que corresponden a diferentes ejecuciones del experimento.
2.  **Localización de Archivos:** Para cada tamaño de dominio, construye la ruta al archivo `model-classifier.json` dentro del directorio de resultados correspondiente (ej. `runs-256/model-classifier.json`).
3.  **Parseo de JSON:** Abre y lee el contenido del archivo JSON.
4.  **Extracción de Métricas:**
    *   Accede a la clave `'history'` del JSON, que contiene una lista con los resultados del entrenamiento y la evaluación.
    *   El script está diseñado para entender la estructura específica de esta lista: sabe que los resultados de la evaluación final para el clasificador y el autoencoder se encuentran en posiciones específicas dentro de la lista.
    *   Extrae dos métricas clave de la evaluación final sobre el conjunto de prueba:
        *   La **precisión (`accuracy`)** del clasificador.
        *   El **error cuadrático medio (`root_mean_squared_error`)** del decodificador (autoencoder).
5.  **Impresión de Resultados:** Imprime en la consola una línea en formato CSV por cada pliegue (fold) de cada configuración de dominio, mostrando: `dominio, precision_clasificador, error_autoencoder`.

### Conclusión

`parse_history.py` es una herramienta de agregación que facilita la revisión del rendimiento de los modelos de redes neuronales. En lugar de tener que abrir manualmente cada archivo JSON, este script extrae y presenta las métricas de rendimiento más importantes de una manera concisa y tabular. Esto es fundamental para verificar la calidad del entrenamiento de los modelos en cada experimento antes de proceder a análisis más complejos que involuven la memoria asociativa.



