# examen_nivel_intermedio_IvesLancelotePerezSanchez
Examen de nivel Intermedio python

# Ejercicio 1 -> Filtrar DataFrame con pandas
Esta función permite ingresar un DataFrame, el nombre de una columna y un umbral. La función accede al DataFrame y toma la columna especificada, luego compara cada valor de esa columna con el umbral, conservando solo las filas donde el valor de la columna es mayor que el umbral. Las demás filas se eliminan.
### Proceso
Filtrar las filas que tengan valores mayores al umbral en la columna especificada.

### Ejemplo de implementación:
- Cargar el dataset bostonHousing.
- Especificar que la primera columna actúa como índice.
- Llamar a la función filter_dataframe con la columna y el umbral deseados para obtener el DataFrame filtrado.

# Ejercicio 2 -> Generar datos para regresión
Esta función permite generar un conjunto de datos simulado para un problema de regresión, con una cantidad de muestras especificada por el usuario. La función crea varias variables independientes y una variable dependiente (con ruido) basada en una combinación lineal de las variables independientes.

### Proceso
- Generar tres variables independientes simuladas usando valores aleatorios.
- Crear la variable dependiente a partir de una combinación lineal de las variables independientes y añadiendo ruido para tener variabilidad.

### Ejemplo de implementación:
- Llamar a la función generate_regression_data con el número deseado de muestras.
- Recibir un conjunto de datos con las variables independientes (X) y la variable dependiente (y), que pueden ser usados para entrenar modelos de regresión.

# Ejercicio 3 -> Entrenar modelo de regresión múltiple
Esta función entrena un modelo de regresión lineal múltiple utilizando un conjunto de datos con variables independientes y una variable dependiente.

### Proceso
- Ajustar un modelo de regresión lineal usando las variables independientes (X) y la variable dependiente (y) proporcionadas.
- Devolver el modelo entrenado que se puede usar para realizar predicciones o evaluar sus coeficientes e intercepto.

### Ejemplo de implementación
- Obtener un modelo de regresión entrenado. . Luego, se pueden calcular los coeficientes (la relación entre cada variable independiente y la dependiente) y el intercepto (el valor de la variable dependiente cuando todas las variables independientes son 0).

# Ejercicio 4 -> List comprehension anidado
Esta función aplana una lista de listas en una lista simple

### Proceso
- Recorrer cada sublista dentro de la lista principal.
- Extraer cada elemento de la sublista y añadirlo a una nueva lista "plana".
- Devolver la lista aplanada como resultado.

### Ejemplo de implementación
- Llamar a la función flatten_list pasando una lista de listas como argumento.
- La función devolverá una lista con todos los elementos de las sublistas en un solo nivel.

# Ejercicio 5 -> Agrupar y agregar con pandas
Esta función agrupa un DataFrame según una columna específica y calcula la media de otra columna. 

### Proceso
- Agrupar el DataFrame utilizando la columna especificada para agrupar.
- Calcular la media de la columna que se desea agregar.
- Reiniciar el índice del DataFrame resultante para que los grupos sean accesibles como filas.

### Implementación:
- Llamar a la función group_and_aggregate pasando un DataFrame, el nombre de la columna por la que se desea agrupar y la columna cuyas medias se quieren calcular.
- La función devolverá un nuevo DataFrame que contiene los valores agrupados y sus promedios correspondientes.

# Ejercicio 6 -> Modelo de lcasificación logística
Esta función entrena un modelo de regresión logística utilizando datos binarios. Es particularmente útil para problemas de clasificación, donde el objetivo es predecir una de dos clases.

### Proceso:
- Crear un modelo de regresión logística.
- Ajustar el modelo a los datos independientes y dependientes proporcionados.
- Devolver el modelo entrenado.

### Ejemplo de implementación
- Cargar un conjunto de datos petAdoption y dividirlo en conjuntos de entrenamiento y prueba.
- Llamar a la función train_logistic_regression con los datos de entrenamiento.
- Utilizar el modelo entrenado para hacer predicciones en el conjunto de prueba.

# Ejercicio 7 -> Aplicar función a una columna con pandas
Esta función aplica una función personalizada a cada valor de una columna específica en un DataFrame de pandas.

### Proceso
- Utilizar el método .apply() de pandas para aplicar la función a cada elemento de la columna especificada.
- Actualizar la columna del DataFrame con los nuevos valores calculados.
- Retornar el DataFrame modificado.

### Ejemplo de implementación:
- Cargar un DataFrame desde un archivo CSV llamado "bostonHousing.csv", utilizando la primera columna como índice.
- Tomar las filas necesarias del DataFrame.
- Llamar a la función apply_function_to_column para aplicar la función cuadrática a la columna en este caso 'zn'.

# Ejercicio 8 -> Aplicar función a una columna con pandas
Esta función filtra una lista de números para conservar solo aquellos que son mayores que cinco y calcula el cuadrado de esos números.

### Proceso
- Crear una lista vacía para almacenar los números cuadrados.
- Iterar sobre cada número en la lista proporcionada.
- Verificar si el número es mayor que cinco; si es así, calcular su cuadrado y añadirlo a la lista de resultados.
- Retornar la lista de números cuadrados que cumplieron con la condición.

### Ejemplo de implementación
- Definir una lista de números.
- Llamar a la función filter_and_square pasando la lista como argumento.
- La función retornará una nueva lista con los números filtrados y elevados al cuadrado.

