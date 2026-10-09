# Actividad 3 - Métodos de aprendizaje supervisado

## Proyecto

Modelo supervisado para predecir retrasos en un sistema de transporte.

El proyecto continúa el sistema de transporte desarrollado en las actividades anteriores.

Se implementó un árbol de decisión utilizando Python estándar. El modelo analiza información de viajes y permite predecir si un recorrido puede presentar retraso.

## Variables utilizadas

- pasajeros
- numero_paradas
- transbordos
- tiempo_estimado
- hora_pico

La variable objetivo es:

- retraso: Si / No

## Dataset

El archivo utilizado es:

datos/datos_transporte.csv

El archivo CSV contiene los registros utilizados para entrenar y evaluar el modelo.

CSV significa Comma-Separated Values o valores separados por comas.

Cada fila representa un viaje y cada columna contiene una característica del recorrido.

## Funcionamiento del modelo

El programa realiza los siguientes pasos:

1. Carga el dataset.
2. Calcula entropía y ganancia de información.
3. Construye un árbol de decisión.
4. Divide los datos en entrenamiento y prueba.
5. Entrena el modelo.
6. Realiza predicciones.
7. Calcula las métricas de evaluación.
8. Muestra las reglas aprendidas.
9. Permite realizar nuevas predicciones.

## Librerías utilizadas

El proyecto utiliza únicamente módulos incluidos en Python:

- csv
- math
- random
- collections
- dataclasses
- pathlib

No es necesario instalar pandas ni scikit-learn.

## Ejecución

Desde la carpeta principal del proyecto ejecutar:

py .\src\modelo_supervisado.py

También puede utilizarse:

python .\src\modelo_supervisado.py

## Estructura del proyecto

Actividad-3---M-todos-de-aprendizaje-supervisado/
│
├── datos/
│   └── datos_transporte.csv
│
├── documentos/
│   ├── Descripcion_Datos.pdf
│   ├── Mapa_Conceptual.pdf
│   └── Pruebas_Modelo.pdf
│
├── src/
│   └── modelo_supervisado.py
│
└── README.md