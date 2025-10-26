# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from pydantic import BaseModel, Field
from typing import List, Optional, Dict # Asegúrate que List, Optional, Dict estén importados

# --- Se importan TODAS las funciones necesarias del procesador ---
from app.processor import (
    procesar_finanzas_personales,
    procesar_finanzas_empresa,
    generar_plan_meta,
    ejecutar_simulacion_escenarios # Nueva función importada
)

app = FastAPI()

# Configuración de CORS (Ajusta la IP del frontend si es necesario)
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

# === ENDPOINTS ===

@app.get("/")
def read_root():
    """ Endpoint de verificación de estado. """
    return {"status": "API de Análisis Financiero está en línea"}


@app.post("/api/v1/analisis/financiero")
async def analizar_archivo_financiero(file: UploadFile = File(...)):
    """
    Recibe un archivo Excel (.xlsx o .xls), detecta si es de finanzas
    personales o de empresa, lo procesa y devuelve un análisis completo.
    """
    # Validar formato
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Solo .xlsx o .xls.")

    try:
        content = await file.read()
        # Lee solo cabeceras para detectar tipo
        df_headers = pd.read_excel(io.BytesIO(content), nrows=0)
        columnas = df_headers.columns.to_list()
    except Exception as e:
        print(f"Error leyendo cabeceras Excel: {e}")
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo Excel: {str(e)}")

    file_stream = io.BytesIO(content) # Stream para pasar a procesadores
    resultado = None

    # Lógica de detección y delegación
    if "id_usuario" in columnas:
        print("Detectado: Archivo Personal.")
        resultado = procesar_finanzas_personales(file_stream)
    elif "empresa_id" in columnas and "concepto" in columnas:
        print("Detectado: Archivo de Empresa.")
        resultado = procesar_finanzas_empresa(file_stream)
    else:
        columnas_encontradas = ", ".join(columnas)
        raise HTTPException(
            status_code=400,
            detail=f"Formato no reconocido. Columnas: {columnas_encontradas}. Se esperaba 'id_usuario' o 'empresa_id'."
        )

    # Manejo centralizado de errores devueltos por los procesadores
    if isinstance(resultado, dict) and "error" in resultado:
        error_detail = resultado["error"]
        status_code = 500
        if isinstance(error_detail, str):
            if "401" in error_detail or "User not found" in error_detail: status_code = 401
            elif "400" in error_detail: status_code = 400
            elif "Faltan columnas" in error_detail: status_code = 400 # Error específico de columnas
        print(f"Error devuelto por procesador (status {status_code}): {error_detail}")
        raise HTTPException(status_code=status_code, detail=error_detail)

    return resultado


# --- Modelo para el endpoint de Metas ---
class MetaRequest(BaseModel):
    meta_usuario: str
    analisis_descriptivo: dict
    analisis_predictivo: dict

@app.post("/api/v1/metas/generar-plan")
async def crear_plan_para_meta(request: MetaRequest):
    """
    Recibe análisis previo y meta del usuario, devuelve plan generado por IA.
    """
    try:
        plan_ia = generar_plan_meta(
            meta=request.meta_usuario,
            descriptivo=request.analisis_descriptivo,
            predictivo=request.analisis_predictivo
        )
        # Manejo de error devuelto por la función
        # Asume que la función devuelve {"recomendacion": "Error..."} en caso de fallo
        if isinstance(plan_ia, dict) and "recomendacion" in plan_ia and isinstance(plan_ia["recomendacion"], str) and ("Error" in plan_ia["recomendacion"] or "error" in plan_ia["recomendacion"]):
             error_detail = plan_ia["recomendacion"]
             status_code = 500
             if "401" in error_detail or "User not found" in error_detail: status_code = 401
             print(f"Error devuelto por generar_plan_meta (status {status_code}): {error_detail}")
             raise HTTPException(status_code=status_code, detail=error_detail)

        return plan_ia # Devuelve {"recomendacion": "Plan..."}

    except HTTPException as httpe:
         raise httpe # Re-lanza errores HTTP manejados
    except Exception as e:
        print(f"Excepción inesperada en /metas/generar-plan: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error inesperado generando el plan: {str(e)}")


# --- Modelos y Endpoint para Simulación ---
class CambioSimulacion(BaseModel):
    tipo: str # 'ingreso' o 'gasto'
    categoria: Optional[str] = None # Requerido si tipo='gasto' y aplica a categoría específica
    # Permitir uno de los dos: porcentaje o monto fijo
    porcentaje_cambio: Optional[float] = None # Ej. 0.10 (10%), -0.05 (-5%)
    monto_fijo_cambio: Optional[float] = None # Ej. 100 ($100 más), -50 ($50 menos)
    # Podrías añadir: mes_inicio, mes_fin, frecuencia ('unico', 'mensual')

class SimulacionRequest(BaseModel):
    analisis_descriptivo: dict # Datos base del análisis original
    meses_a_proyectar: int = Field(..., gt=0, le=60) # Proyectar entre 1 y 60 meses (5 años)
    cambios: List[CambioSimulacion] = [] # Lista (puede ser vacía) de cambios a simular

@app.post("/api/v1/simulacion/proyectar")
async def proyectar_escenarios(request: SimulacionRequest):
    """
    Ejecuta una simulación financiera "What-If" basada en datos previos
    y cambios especificados, devolviendo una proyección de balance.
    """
    try:
        resultado_simulacion = ejecutar_simulacion_escenarios(
            datos_base=request.analisis_descriptivo,
            meses=request.meses_a_proyectar,
            cambios_escenario=request.cambios # Pasa la lista de objetos Pydantic
        )

        if isinstance(resultado_simulacion, dict) and "error" in resultado_simulacion:
            # Puedes ser más específico con el código de error si la simulación falla por datos inválidos
            raise HTTPException(status_code=400, detail=resultado_simulacion["error"])

        # Devuelve el JSON: {"fechas": [...], "proyeccion": [...]}
        return resultado_simulacion

    except HTTPException as httpe:
        raise httpe
    except Exception as e:
        print(f"Error inesperado en /simulacion/proyectar: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error inesperado ejecutando la simulación: {str(e)}")

# --- FIN DEL ARCHIVO app/main.py ---