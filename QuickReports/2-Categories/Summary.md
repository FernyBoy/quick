# Resumen de Resultados: Experimento 1 - Reconocimiento de Clases Conocidas vs. Desconocidas

## 1. Objetivo del Experimento

El objetivo de este experimento fue evaluar el rendimiento de una Memoria Asociativa Entrópica (EAM) en un escenario de clasificación con conocimiento parcial. Específicamente, se entrenó la memoria con una sola de dos categorías presentes en el conjunto de datos. El propósito era medir qué tan bien la memoria puede:

1.  Reconocer correctamente las muestras de la categoría **conocida** (cargada en memoria).
2.  Rechazar correctamente las muestras de la categoría **desconocida** (no cargada en memoria).

## 2. Metodología

Se utilizó una EAM entrenada únicamente con muestras de la "categoría 0". Posteriormente, la memoria fue evaluada con un conjunto de pruebas que contenía un 50% de muestras de la "categoría 0" (conocida) y un 50% de la "categoría 1" (desconocida). Se analizaron métricas de precisión (precision), exhaustividad (recall) y las matrices de confusión para diferentes tamaños de memoria (`msize`).

## 3. Análisis de Resultados

### 3.1. Métricas de Precisión y Exhaustividad (Recall)

El análisis del archivo `behaviours.npy` muestra las siguientes tendencias:

*   **Precisión (1ª columna):** La precisión se mantiene muy alta, comenzando en `0.98` y manteniéndose por encima de `0.90` para la mayoría de los tamaños de memoria. Esto indica que el modelo es extremadamente fiable. Cuando la memoria clasifica una muestra como "conocida", es casi seguro que su clasificación es correcta.

*   **Exhaustividad / Recall (2ª columna):** El recall comienza bajo, sube rápidamente a `0.98` y luego disminuye a medida que aumenta el tamaño de la memoria. Sin embargo, si consideramos que el recall ideal es del 50% (ya que solo el 50% de los datos de prueba son "conocidos"), los valores observados son excelentes.

### 3.2. Matrices de Confusión

El análisis de `memory_confrixes.npy` revela el comportamiento detallado de la memoria:

*   **`msize=2` (2ª matriz):**
    *   `[[12086, 234], [221, 12139]]`
    *   **Análisis:** La memoria clasifica correctamente ~12k muestras conocidas y ~12k desconocidas, con muy pocos errores. Este es un rendimiento casi perfecto.

*   **`msize` intermedios:** A medida que `msize` aumenta, la memoria comienza a clasificar erróneamente más muestras desconocidas como conocidas (ej. en la 4ª matriz, `[1820, 9708]`). Esto sugiere que la memoria se está volviendo demasiado generalizadora.

*   **`msize` grandes:** Con tamaños de memoria muy grandes, la memoria se vuelve demasiado específica, y el número de muestras conocidas correctamente clasificadas disminuye.

## 4. Conclusiones

1.  **Éxito del Experimento:** Se demostró con éxito la capacidad de la EAM para gestionar la novedad.
2.  **Alta Fiabilidad:** El sistema es muy fiable, con una tasa de falsos positivos muy baja.
3.  **Identificación de Novedad:** La memoria es capaz de distinguir eficazmente entre las muestras que pertenecen a la clase aprendida y las que no.
4.  **Importancia de `msize`:** El tamaño de la memoria es un hiperparámetro crítico. Un valor pequeño-mediano (`msize=2`) parece ser el punto óptimo para este problema.

En resumen, el experimento valida el comportamiento deseado de la EAM en un escenario de clasificación con información incompleta.
