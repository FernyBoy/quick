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
