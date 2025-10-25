from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict 

# Importa tu nueva clase
from .processor import DataProcessor

# --- Configuración de IA y Variables Globales ---

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

IA_CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Servidor MCP Financiero"
    }
)

PROCESADOR_ACTUAL: DataProcessor = None

class FullSimulationRequest(BaseModel):
    # Un diccionario donde la llave es la categoría (string) y el valor es el nuevo monto (float)
    gastos_simulados: Dict[str, float] = Field(..., example={"Restaurantes": 800, "Transporte": 750, "Renta": 4000})


# El modelo para /analizar sigue igual (simple)
class AnalysisRequest(BaseModel):
    meta_financiera: str

# --- Creación de la App ---
app = FastAPI(title="Servidor MCP Financiero")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints del API ---

@app.post("/upload")
async def upload_financial_data(file: UploadFile = File(...)):
    """
    ESTE ENDPOINT ES UTILIZADO PARA LA LECTURA Y PROCESAMIENTO DE LOS ARCHIVOS CSV O XLSX
    """
    global PROCESADOR_ACTUAL
    filename = file.filename
    if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Por favor sube un archivo .csv o .xlsx")
    file_content = await file.read()
    try:
        PROCESADOR_ACTUAL = DataProcessor(file_content, filename) 
        # Esta función ahora devuelve el "reporte_gastos_actual"
        initial_data = PROCESADOR_ACTUAL.get_initial_data() 
        return {
            "message": f"Archivo '{filename}' procesado exitosamente.",
            "initial_data": initial_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {e}")


@app.post("/simular", tags=["Simulación Rápida"])
async def post_simulation(request: FullSimulationRequest): # <-- ¡MODELO ACTUALIZADO!
    """
        Muestra al usuario como cambia su proyección de ahorro futuro si modifica sus habitos de gasto
    """
    if PROCESADOR_ACTUAL is None:
        raise HTTPException(status_code=400, detail="Sube un archivo primero en el endpoint /upload")
        
    # Llamamos a la nueva función del procesador
    sim_data = PROCESADOR_ACTUAL.get_simulation_full_report(request.gastos_simulados)
    return sim_data


@app.post("/analizar", tags=["Análisis con IA"])
async def post_analysis(request: AnalysisRequest):
    """
    EL USUARIO INGRESA UNA META FINANCIERA Y EL MODELO DE IA TE DA UNA RECOMENDACIÓN PERSONALIZADA
    EN BASE A TUS REGISTROS FINANCIEROS
    """
    if PROCESADOR_ACTUAL is None:
        raise HTTPException(status_code=400, detail="Sube un archivo primero en el endpoint /upload")

    # 1. Obtenemos los datos TOTALES del procesador
    ingresos_reales = PROCESADOR_ACTUAL.ingreso_promedio_calculado
    gastos_reales = PROCESADOR_ACTUAL.gasto_promedio_calculado

    # 2. Obtenemos el REPORTE COMPLETO
    resumen_detallado_dict = PROCESADOR_ACTUAL.resumen_gastos_por_categoria

    # 3. Creamos el contexto general
    contexto_usuario = (
        f"El usuario tiene ingresos mensuales promedio de ${ingresos_reales:,.2f} "
        f"y gastos mensuales promedio de ${gastos_reales:,.2f}. "
        f"Su meta financiera es: '{request.meta_financiera}'."
    )
    
    # 4. Formateamos el REPORTE COMPLETO para el prompt
    reporte_string = "\n".join([
        f"- {categoria}: ${monto:,.2f} (promedio mensual)" 
        for categoria, monto in resumen_detallado_dict.items()
    ])

    # 5. Creamos el PROMPT MEJORADO
    prompt = (
        "Eres un asesor financiero experto y analítico. Un usuario te comparte su situación: "
        f"'{contexto_usuario}'.\n\n"
        "Además, aquí está el desglose de sus gastos promedio mensuales (el 'reporte completo' basado en su archivo):\n"
        f"{reporte_string}\n\n"
        "Basándote **específicamente en todos estos números y en el desglose de gastos**,"
        "dame 3 recomendaciones accionables y **cuantificadas** para alcanzar su meta. "
        "Sé específico; si ves un gasto alto en una categoría notable (ej. 'Restaurantes', 'Compras'), coméntalo. "
        "No des consejos genéricos. Habla en segunda persona (tú). Sé breve."
    )
    
    try:
        response = IA_CLIENT.chat.completions.create(
            model="openai/gpt-4", 
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        recomendacion = response.choices[0].message.content
        return {"recomendacion": recomendacion}
    
    except Exception as e:
        return {"recomendacion": f"Error al contactar la IA (OpenRouter): {str(e)}"}