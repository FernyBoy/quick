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
