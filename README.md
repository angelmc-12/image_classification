# MNIST Classifier From Scratch (CNN) + Streamlit Demo

Proyecto de portafolio: entrené una red neuronal convolucional desde cero para clasificar dígitos (0–9) y construí una demo interactiva con Streamlit para probar el modelo.

## Demo


## ¿Qué incluye?
- Entrenamiento de una CNN sencilla en PyTorch
- Guardado de pesos del modelo en `models/`
- Evaluación en validación + matriz de confusión
- App Streamlit para dibujar/subir imágenes y ver predicciones + probabilidades

## Cómo correrlo

### 1) Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2) Entrenar el modelo

```bash
python -m src.train --epochs 8 --lr 0.05
```

Genera:

- `models/mnist_cnn.pt`
- `assets/training_curves.png`
- `assets/confusion_matrix.png`

### 3) Ejecutar la demo

```bash
streamlit run app/streamlit_app.py
```

## Resultados

- Accuracy (validación): (completar)
- Matriz de confusión: (completar)
