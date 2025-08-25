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
- **Acción:** Esta es una de las fases más críticas. Realiza una búsqueda sistemática para encontrar los mejores hiperparámetros para la memoria asociativa. Se entrenan y evalúan **todas** las clases especificadas.
    - **`test_memory_sizes`**: Itera sobre una lista de posibles tamaños de memoria (`constants.memory_sizes`). Para cada tamaño, entrena la memoria y mide su rendimiento (precisión, recall, entropía) usando validación cruzada.
    - **`test_memory_fills`**: Para los tamaños de memoria que resultaron ser óptimos, analiza cómo varía el rendimiento a medida que la memoria se llena con diferentes porcentajes del corpus de datos (ej. 10%, 20%, ..., 100%).
    - **`save_learned_params`**: Al final, guarda los mejores parámetros encontrados (los tamaños de memoria y porcentajes de llenado más eficientes) en un archivo para su uso en las siguientes fases.

### 4. `-e 2`: Prueba de Detección de Novedad
- **Función:** `run_evaluation()`
- **Acción:** Este experimento prueba la capacidad de la memoria para rechazar patrones de clases que no conoce.
    - **Carga Parcial:** Deliberadamente, solo se entrena la memoria con la **mitad** de las clases disponibles.
    - **Evaluación Completa:** Se evalúa el rendimiento contra el conjunto de prueba completo, que incluye tanto las clases conocidas como las desconocidas.
    - **Objetivo:** Medir qué tan efectivamente la memoria rechaza los patrones de las clases que nunca ha visto, evaluando su robustez frente a datos novedosos.

### 5. `-r`: Generación de Recuerdos (Visualización)
- **Función:** `generate_memories()`
- **Acción:** Carga los mejores parámetros de memoria encontrados en la fase de evaluación (`-e`).
    1.  Crea una memoria con esa configuración óptima y la llena con los datos de entrenamiento.
    2.  Le presenta a la memoria los datos de prueba (tanto los originales como versiones con ruido) y le pide que "recuerde" los patrones (`remember()`).
    3.  Los patrones recordados (vectores de características) se pasan a través del **decodificador** (`decode_memories()`) para reconstruir las imágenes.
    4.  Guarda estas imágenes reconstruidas, permitiendo visualizar qué ha recuperado la memoria y comparar la calidad del recuerdo para datos limpios y con ruido.

### 6. `-d`: Proceso de "Ensoñación" (Dreaming)
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
