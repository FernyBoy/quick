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
