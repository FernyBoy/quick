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
