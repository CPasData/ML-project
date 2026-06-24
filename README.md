# 🎮 Predicción de Ventas de Videojuegos

Proyecto de Machine Learning para predecir las ventas globales de videojuegos a partir de características del juego como plataforma, género, puntuaciones de crítica y usuarios, publisher y año de lanzamiento.

---

## 📁 Estructura del proyecto

```
├── data/
│   ├── raw/                  # Dataset original
│   ├── train/                # Sets de entrenamiento sin procesar
│   ├── test/                 # Sets de test sin procesar
│   └── processed/            # Datos tras limpieza y feature engineering
├── models/                   # Modelos entrenados serializados (.pkl)
├── notebooks/
│   ├── 01_fuentes.ipynb      # Carga y exploración inicial del dataset
│   ├── 02_LimpiezaEDA.ipynb  # Limpieza, EDA y feature engineering
│   └── 03_Entrenamiento_Evaluacion.ipynb  # Entrenamiento, evaluación y demo
└── README.md
└── main.py
└── pyproject.toml
└── uv.lock
└── .gitignore
└── .python-version
```

---

## 📊 Dataset

**Fuente:** `Video_Games_Sales_as_at_22_Dec_2016.csv`

El dataset contiene información de ventas de videojuegos con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `Rank` | Ranking de ventas globales |
| `Name` | Nombre del videojuego |
| `Platform` | Plataforma (PS4, PC, Wii, etc.) |
| `Year_of_Release` | Año de lanzamiento |
| `Genre` | Género del juego |
| `Publisher` | Empresa publicadora |
| `NA_Sales` | Ventas en Norteamérica (millones) |
| `EU_Sales` | Ventas en Europa (millones) |
| `JP_Sales` | Ventas en Japón (millones) |
| `Other_Sales` | Ventas en el resto del mundo (millones) |
| `Global_Sales` | Ventas globales totales (millones) — **target** |
| `Critic_Score` | Puntuación agregada de críticos (Metacritic) |
| `Critic_Count` | Número de críticos |
| `User_Score` | Puntuación de usuarios (Metacritic) |
| `User_Count` | Número de usuarios que puntuaron |
| `Developer` | Desarrollador del juego |
| `Rating` | Clasificación ESRB |

---

## 🔄 Pipeline del proyecto

### Notebook 1 — Fuentes (`01_fuentes.ipynb`)
- Carga del dataset con `pandas`
- Inspección inicial: tipos de datos, estadísticas descriptivas y estructura general

### Notebook 2 — Limpieza y EDA (`02_LimpiezaEDA.ipynb`)

**Limpieza:**
- Split train/test (80/20) con `random_state=42`
- Eliminación de valores nulos (alta concentración de missings en columnas de crítica para años anteriores a 1995)

**Análisis exploratorio:**
- El target `Global_Sales` presenta fuerte asimetría positiva: la mayoría de juegos venden menos de 1M de unidades. Se aplica transformación logarítmica (`log1p`) para normalizar la distribución.
- Ventas globales por género, plataforma, publisher y región.
- Pico máximo de ventas globales en 2008 (~400M de unidades), con posterior descenso.
- Norteamérica es la región con mayor media de ventas.
- Alta colinealidad entre ventas regionales y ventas globales (esperado).

**Feature Engineering:**
- `Decade`: categorización del año de lanzamiento en décadas (`pre-90s`, `90s`, `2000s`, `2010s`)
- `Cat_Publisher`: agrupación de publishers por número de juegos publicados en 3 categorías (evita alta cardinalidad)
- Eliminación de columnas: `Name`, `Year_of_Release`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Publisher`, `Developer`
- One Hot Encoding sobre columnas categóricas: `Genre`, `Platform`, `Rating`, `Decade`, `Cat_Publisher`
- Target transformado con `log1p` y guardado en `data/processed/`

> **Nota:** Se exploró la adición de clusters via KMeans sobre variables numéricas (método del codo sugería k=4), pero finalmente fue descartada del pipeline final.

### Notebook 3 — Entrenamiento y Evaluación (`03_Entrenamiento_Evaluacion.ipynb`)

**Modelos evaluados:**
- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor

**Búsqueda de hiperparámetros:**
- `GridSearchCV` (búsqueda exhaustiva, cv=3, scoring=`neg_mean_absolute_error`)
- `RandomizedSearchCV` (100 iteraciones, cv=3) como alternativa más eficiente

Cada modelo se envuelve en un `Pipeline` de scikit-learn con escalado opcional (`StandardScaler`, `MinMaxScaler` o sin escalado).

**Métricas de evaluación** (sobre valores reales, tras aplicar `expm1`):
- MAE — Error Absoluto Medio
- MAPE — Error Porcentual Absoluto Medio
- MSE — Error Cuadrático Medio
- RMSE — Raíz del Error Cuadrático Medio
- R² — Coeficiente de determinación

**Modelo final:** XGBoost Regressor (seleccionado automáticamente por GridSearchCV), guardado como `models/OHE_drop.pkl`.

---

## 🚀 Demo de predicción

El notebook 3 incluye una demo que genera una muestra aleatoria con características simuladas y predice las ventas globales esperadas:

```python
prediccion_log = modelo_importado.predict(sample)
prediccion_real = np.expm1(prediccion_log)
print(f"Ventas predichas: {prediccion_real[0]:.2f} millones de unidades")
```

---

## 🛠️ Tecnologías utilizadas

- **Python 3**
- `pandas`, `numpy` — manipulación de datos
- `matplotlib`, `seaborn` — visualización
- `scikit-learn` — preprocesamiento, pipelines, modelos y evaluación
- `xgboost` — modelo de gradient boosting
- `scipy` — estadística
- `pickle` — serialización del modelo

---

## ▶️ Cómo ejecutar

1. Clona el repositorio e instala las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
   ```

2. Coloca el dataset original en `data/raw/Video_Games_Sales_as_at_22_Dec_2016.csv`

3. Ejecuta los notebooks en orden:
   ```
   01_fuentes.ipynb → 02_LimpiezaEDA.ipynb → 03_Entrenamiento_Evaluacion.ipynb
   ```
