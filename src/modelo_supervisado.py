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


