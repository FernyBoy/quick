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
