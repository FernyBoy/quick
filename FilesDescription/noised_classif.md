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
