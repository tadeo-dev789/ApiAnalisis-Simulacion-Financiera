# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# --- Se importan las dos funciones de procesamiento ---
from app.processor import procesar_finanzas_personales, procesar_finanzas_empresa

# --- Se importan Pydantic y la función para metas ---
from pydantic import BaseModel
from app.processor import generar_plan_meta


app = FastAPI()

# Tu configuración de CORS (Asegúrate que la IP del frontend esté correcta)
# Ejemplo: "http://<IP_PUBLICA_DEL_FRONTEND>"
origins = [
    "http://141.148.66.152", # Reemplaza con la IP real de tu Frontend
    "http://localhost",
    "http://localhost:5173", # Puerto de desarrollo de Vite
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API de Análisis Financiero está en línea"}


@app.post("/api/v1/analisis/financiero")
async def analizar_archivo_financiero(file: UploadFile = File(...)):

    # Validar que sea un archivo de Excel
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Por favor, sube un archivo .xlsx o .xls.")

    try:
        # Leemos el archivo en memoria
        content = await file.read()

        # Usamos 'nrows=0' para leer SÓLO las cabeceras
        df_headers = pd.read_excel(io.BytesIO(content), nrows=0)
        columnas = df_headers.columns.to_list()

    except Exception as e:
        # Imprime el error real si falla la lectura del Excel
        print(f"Error leyendo cabeceras Excel: {e}")
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo Excel. Error: {str(e)}")

    # --- LÓGICA DE DETECCIÓN ---
    # Creamos un stream (BytesIO) para pasar el contenido a las funciones de procesamiento
    file_stream = io.BytesIO(content)

    resultado = None # Inicializa resultado

    # Revisa si tiene columnas de "Personal"
    if "id_usuario" in columnas:
        print("Detectado: Archivo Personal. Iniciando procesamiento...")
        resultado = procesar_finanzas_personales(file_stream)

    # Revisa si tiene columnas de "Empresa"
    elif "empresa_id" in columnas and "concepto" in columnas:
        print("Detectado: Archivo de Empresa. Iniciando procesamiento...")
        resultado = procesar_finanzas_empresa(file_stream)

    # Si no es ninguno
    else:
        columnas_encontradas = ", ".join(columnas)
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no reconocido. Columnas encontradas: {columnas_encontradas}. Se esperaba 'id_usuario' o 'empresa_id'."
        )

    # Si el procesamiento tuvo un error interno (como el 401 de OpenAI)
    if isinstance(resultado, dict) and "error" in resultado:
        error_detail = resultado["error"]
        print(f"Error devuelto por el procesador: {error_detail}")
        status_code = 500 # Default a error interno
        if "401" in error_detail:
             status_code = 401 # Específico para errores de autenticación (API Key)
        elif "400" in error_detail: # Podrías añadir más códigos si los necesitas
             status_code = 400
        raise HTTPException(status_code=status_code, detail=error_detail)

    # ¡Todo salió bien!
    return resultado


# --- Endpoint para Metas ---
class MetaRequest(BaseModel):
    meta_usuario: str
    analisis_descriptivo: dict
    analisis_predictivo: dict

@app.post("/api/v1/metas/generar-plan")
async def crear_plan_para_meta(request: MetaRequest):
    """
    Recibe los datos de un análisis previo y una meta del usuario,
    y devuelve un plan generado por IA.
    """
    try:
        # Llama al especialista en el procesador
        plan_ia = generar_plan_meta(
            meta=request.meta_usuario,
            descriptivo=request.analisis_descriptivo,
            predictivo=request.analisis_predictivo
        )

        # Revisa si la función generar_plan_meta devolvió un error
        if isinstance(plan_ia, dict) and "error" in plan_ia:
             error_detail = plan_ia["error"]
             print(f"Error devuelto por generar_plan_meta: {error_detail}")
             status_code = 500
             if "401" in error_detail:
                  status_code = 401
             raise HTTPException(status_code=status_code, detail=error_detail)

        return plan_ia

    except Exception as e:
        # Captura cualquier otra excepción inesperada
        print(f"Excepción inesperada en /metas/generar-plan: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado generando el plan: {str(e)}")