import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==========================================
# 1. CARGAR LOS DATOS
# ==========================================

datos = pd.read_csv(
    "datos_transporte.csv"
)

print("\n======================================")
print("DATOS DEL SISTEMA DE TRANSPORTE")
print("======================================")

print(datos.head())


# ==========================================
# 2. VARIABLES DE ENTRADA
# ==========================================

X = datos[
    [
        "pasajeros",
        "numero_paradas",
        "transbordos",
        "tiempo_estimado",
        "hora_pico"
    ]
]


# ==========================================
# 3. VARIABLE OBJETIVO
# ==========================================

y = datos["retraso"]


# ==========================================
# 4. DIVIDIR ENTRENAMIENTO Y PRUEBA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\nDatos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))


# ==========================================
# 5. CREAR ÁRBOL DE DECISIÓN
# ==========================================

modelo = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# ==========================================
# 6. ENTRENAR EL MODELO
# ==========================================

modelo.fit(
    X_train,
    y_train
)


# ==========================================
# 7. REALIZAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(
    X_test
)


# ==========================================
# 8. PRECISIÓN DEL MODELO
# ==========================================

precision = accuracy_score(
    y_test,
    predicciones
)

print("\n======================================")
print("RESULTADOS DEL MODELO")
print("======================================")

print(
    f"\nPrecisión: {precision * 100:.2f}%"
)


# ==========================================
# 9. MATRIZ DE CONFUSIÓN
# ==========================================

matriz = confusion_matrix(
    y_test,
    predicciones
)

print("\nMatriz de confusión:")

print(matriz)


# ==========================================
# 10. REPORTE DE CLASIFICACIÓN
# ==========================================

print("\nReporte de clasificación:")

print(
    classification_report(
        y_test,
        predicciones,
        zero_division=0
    )
)


# ==========================================
# 11. REGLAS APRENDIDAS
# ==========================================

reglas = export_text(
    modelo,
    feature_names=list(X.columns)
)

print("\n======================================")
print("REGLAS APRENDIDAS POR EL ÁRBOL")
print("======================================")

print(reglas)


# ==========================================
# 12. NUEVA PREDICCIÓN
# ==========================================

print("\n======================================")
print("NUEVA PREDICCIÓN")
print("======================================")


pasajeros = int(
    input(
        "\nCantidad de pasajeros: "
    )
)

numero_paradas = int(
    input(
        "Número de paradas: "
    )
)

transbordos = int(
    input(
        "Número de transbordos: "
    )
)

tiempo_estimado = int(
    input(
        "Tiempo estimado en minutos: "
    )
)


while True:

    hora_pico = int(
        input(
            "¿Es hora pico? "
            "1 = Sí / 0 = No: "
        )
    )

    if hora_pico in [0, 1]:
        break

    print(
        "Ingrese únicamente 1 o 0."
    )


nuevo_viaje = pd.DataFrame(
    [
        {
            "pasajeros": pasajeros,
            "numero_paradas": numero_paradas,
            "transbordos": transbordos,
            "tiempo_estimado": tiempo_estimado,
            "hora_pico": hora_pico
        }
    ]
)


resultado = modelo.predict(
    nuevo_viaje
)


print("\n======================================")
print("PREDICCIÓN DEL SISTEMA")
print("======================================")

print(
    f"\nResultado: {resultado[0]}"
)


if resultado[0] == "Si":

    print(
        "El sistema predice que "
        "el viaje puede presentar retraso."
    )

else:

    print(
        "El sistema predice que "
        "el viaje llegará a tiempo."
    )