# Servidor Financiero (API de Análisis y Simulación)

Este proyecto es el backend de una aplicación de finanzas personales, construido con **FastAPI**.

La API permite a los usuarios subir su historial de transacciones (en `.csv` o `.xlsx`) y realiza dos funciones principales:
1.  **Análisis con IA:** Proporciona recomendaciones financieras personalizadas basadas en los gastos e ingresos reales del usuario, utilizando un modelo de IA (GPT-4 vía OpenRouter).
2.  **Simulación Visual:** Permite al usuario simular cambios en *todos* sus hábitos de gasto y ver visualmente cómo impactaría su proyección de ahorro futuro.

---

## 🚀 Stack Tecnológico

* **FastAPI**: Para el servidor API de alto rendimiento.
* **Uvicorn**: Como servidor ASGI para ejecutar la API.
* **Pandas**: Para la ingesta, limpieza y análisis de los archivos de transacciones.
* **Scikit-learn**: Para crear el modelo de regresión lineal que genera la proyección de ahorro base.
* **OpenAI (SDK)**: Como cliente para conectarse a la API de OpenRouter.
* **Python-dotenv**: Para gestionar las variables de entorno (`.env`).
* **Openpyxl**: Para permitir la lectura de archivos `.xlsx`.

---

## ⚙️ Instalación

Sigue estos pasos para levantar el servidor en tu máquina local.

1.  **Clona este repositorio** (o asegúrate de estar en la carpeta raíz `backend-mcp`).

2.  **Crea y activa un entorno virtual:**
    ```bash
    # Crear el entorno
    python -m venv venv

    # Activar en Windows
    .\venv\Scripts\activate

    # Activar en macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔑 Configuración

Este proyecto requiere una clave de API para conectarse al servicio de IA.

1.  Crea un archivo llamado `.env` en la raíz del proyecto (`backend-mcp/.env`).
2.  Añade tu clave de API de OpenRouter al archivo:

    ```ini
    # Reemplaza "sk-or-v1..." con tu clave real
    OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
    ```

---

## ▶️ Ejecución

Una vez instaladas las dependencias y configurado el `.env`, puedes iniciar el servidor:

```bash
uvicorn app.main:app --reload
```

* `app.main`: Busca el archivo `main.py` dentro de la carpeta `app/`.
* `:app`: Busca la variable `app = FastAPI(...)` dentro de ese archivo.
* `--reload`: Reinicia el servidor automáticamente cada vez que guardas cambios en el código.

Tu API estará disponible en `http://127.0.0.1:8000`.
Puedes ver la documentación interactiva (y probar los endpoints) en: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📖 Flujo de la API (Uso)

El servidor utiliza una variable global (`PROCESADOR_ACTUAL`) que actúa como un "cerebro" en memoria. Por lo tanto, el orden de las llamadas a la API es importante.

### Paso 1: `POST /upload` (Cargar Archivo)

Este es el **primer endpoint** que se debe llamar.