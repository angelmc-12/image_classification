# MNIST Classifier — Demo amigable

Este proyecto contiene un modelo sencillo de red neuronal (CNN) que reconoce dígitos escritos a mano (0–9) y una demo interactiva para probarlo desde el navegador.

No necesitas ser programador para usar la demo: sigue los pasos de "Probar la demo" más abajo. Si quieres entrenar el modelo o modificar código, hay instrucciones para desarrolladores también.

**¿Qué hace esta demo?**
- Permite dibujar o subir una imagen de un dígito y muestra la predicción (0–9) más probable y las probabilidades por clase.
- Incluye pesos de un modelo ya entrenado en `models/mnist_cnn.pt` (si están presentes).

**Archivos importantes**
- `app/streamlit_app.py`: la aplicación web (usa Streamlit).
- `src/`: código fuente (preprocesado, modelo, entrenamiento, predicción).
- `models/`: aquí van los pesos guardados (`mnist_cnn.pt`).

**Probar la demo (rápido, en tu máquina)**
1. Abre una terminal y sitúate en la carpeta del proyecto:

```bash
cd /ruta/al/proyecto/image_classification
```

2. Activa tu entorno Python (si usas conda):

```bash
conda activate tfmac
```

3. Instala dependencias (si aún no lo hiciste):

```bash
pip install -r requirements.txt
# si no existe requirements.txt instala al menos:
pip install streamlit torch pillow numpy
```

4. (Opcional) Para poder dibujar directamente en la demo instala el componente de dibujo:

```bash
pip install streamlit-drawable-canvas
```

5. Ejecuta la demo:

```bash
streamlit run app/streamlit_app.py
```

6. Abre el enlace que aparece en la terminal (normalmente http://localhost:8501) en tu navegador.

Cómo usar la página web:
- Elige "🖊️ Dibujar" para dibujar un dígito (si tienes instalado el componente de dibujo) o "🖼️ Subir imagen" para subir un archivo.
- El modelo mostrará la predicción y una gráfica con las probabilidades por clase.

Solución rápida de problemas
- Error "No module named 'src'": asegúrate de ejecutar Streamlit desde la raíz del repo (`cd image_classification`) y de tener `sys.path` correcto. La app ya incluye una corrección para esto.
- Error "No module named streamlit_drawable_canvas": instala `streamlit-drawable-canvas` o usa el modo de subir imagen (fallback).
- Si falta `models/mnist_cnn.pt`: puedes entrenar el modelo (ver abajo) o pedir el archivo al mantenedor.

Entrenar el modelo (para usuarios con experiencia)
- Si quieres generar los pesos desde cero:

```bash
python -m src.train --epochs 8 --lr 0.05
```

Esto guardará `models/mnist_cnn.pt` y generará imágenes de curvas de entrenamiento en `assets/`.

Contacto y siguientes pasos
- Si necesitas que prepare un instalador, un `requirements.txt` completo o ejecute la demo en un servidor, dímelo y lo preparo.

Gracias — disfruta probando la demo.
