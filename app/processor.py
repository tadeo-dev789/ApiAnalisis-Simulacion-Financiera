# app/processor.py

import pandas as pd
from pydantic import BaseModel # <--- ¡AÑADE ESTA LÍNEA!
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta # Para la simulación
from typing import List, Dict, Optional # Para type hints
# Importa el modelo Pydantic definido en main.py para usarlo aquí
# (Esto asume una estructura de proyecto estándar donde puedes importar desde main)
# Si da error de importación circular, define CambioSimulacion aquí también o en un archivo models.py
# from app.main import CambioSimulacion # Podría causar importación circular, ten cuidado

# Alternativa: Redefinir un TypedDict o Pydantic model aquí si la importación falla
class CambioSimulacionModel(BaseModel): # Usando Pydantic aquí también
    tipo: str
    categoria: Optional[str] = None
    porcentaje_cambio: Optional[float] = None
    monto_fijo_cambio: Optional[float] = None


# --- LÓGICA DE CARGA DE .ENV ---
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
dotenv_loaded = load_dotenv(dotenv_path=env_path, verbose=True)
if not dotenv_loaded:
    print(f"ADVERTENCIA: No se pudo cargar el archivo .env en: {env_path}")
# ---------------------------------

# --- CONFIGURACIÓN PARA OPENROUTER ---
API_KEY_CARGADA = os.environ.get("OPENROUTER_API_KEY")
IA_CLIENT = None
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "http://localhost:5173" # O tu URL
OPENROUTER_APP_TITLE = "Banorte Analisis Financiero"

if not API_KEY_CARGADA:
    print("ERROR FATAL: OPENROUTER_API_KEY no encontrada en el entorno.")
else:
    try:
        IA_CLIENT = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=API_KEY_CARGADA,
            default_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_APP_TITLE,
            },
        )
        print(f"Cliente OpenAI inicializado CORRECTAMENTE para OpenRouter ({OPENROUTER_BASE_URL}).")
    except Exception as e:
        print(f"ERROR: Falló inicialización del cliente OpenAI para OpenRouter: {type(e).__name__} - {e}")
        IA_CLIENT = None

# === FUNCIONES DE LIMPIEZA ===
# (Sin cambios respecto a la versión anterior completa)
def clean_data_personal(df):
    try:
        column_mapping = {'id_usuario': 'id', 'monto': 'monto', 'tipo': 'tipo', 'categoria': 'categoria', 'descripcion': 'descripcion', 'fecha': 'fecha'}
        columnas_requeridas = ['id_usuario', 'monto', 'tipo', 'fecha']
        if not all(col in df.columns for col in columnas_requeridas):
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            return {"error": f"Faltan columnas requeridas (personal): {', '.join(faltantes)}"}

        df = df.rename(columns=column_mapping)
        df = df[[col for col in column_mapping.values() if col in df.columns]]
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['monto'] = pd.to_numeric(df['monto'], errors='coerce').fillna(0)
        df = df.dropna(subset=['fecha', 'monto', 'tipo'])
        df['categoria'] = df['categoria'].astype(str).fillna('Desconocida').str.strip()
        df['descripcion'] = df['descripcion'].astype(str).fillna('Sin descripción').str.strip()
        df['tipo'] = df['tipo'].astype(str).str.lower().str.strip()
        df = df[df['tipo'].isin(['ingreso', 'gasto'])]
        return df
    except Exception as e:
        print(f"Error en clean_data_personal: {e}")
        return {"error": f"Error limpiando datos personales: {str(e)}"}

def clean_data_empresa(df):
    try:
        column_mapping = {'empresa_id': 'id', 'fecha': 'fecha', 'tipo': 'tipo', 'concepto': 'descripcion', 'categoria': 'categoria', 'monto': 'monto'}
        columnas_requeridas = ['empresa_id', 'fecha', 'tipo', 'monto']
        if not all(col in df.columns for col in columnas_requeridas):
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            return {"error": f"Faltan columnas requeridas (empresa): {', '.join(faltantes)}"}

        columnas_presentes = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=columnas_presentes)
        columnas_finales = [v for v in column_mapping.values() if v in df.columns]
        df = df[columnas_finales]
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['monto'] = pd.to_numeric(df['monto'], errors='coerce').fillna(0)
        df = df.dropna(subset=['fecha', 'monto', 'tipo'])
        df['categoria'] = df['categoria'].astype(str).fillna('Desconocida').str.strip()
        df['descripcion'] = df['descripcion'].astype(str).fillna('Sin descripción').str.strip()
        df['tipo'] = df['tipo'].astype(str).str.lower().str.strip()
        df = df[df['tipo'].isin(['ingreso', 'gasto'])]
        return df
    except Exception as e:
        print(f"Error en clean_data_empresa: {e}")
        return {"error": f"Error limpiando datos de empresa: {str(e)}"}

# === FUNCIONES DE ANÁLISIS ===
# (Sin cambios respecto a la versión anterior completa)
def analisis_descriptivo(df):
    try:
        df['tipo'] = df['tipo'].astype(str)
        total_ingresos = df.loc[df['tipo'] == 'ingreso', 'monto'].sum()
        total_gastos = df.loc[df['tipo'] == 'gasto', 'monto'].sum()
        balance = total_ingresos - total_gastos
        gastos_por_categoria = df.loc[df['tipo'] == 'gasto'].groupby('categoria')['monto'].sum()
        gastos_por_categoria = gastos_por_categoria[np.isfinite(gastos_por_categoria)].sort_values(ascending=False).round(2)
        return { "total_ingresos": round(total_ingresos, 2), "total_gastos": round(total_gastos, 2), "balance_neto": round(balance, 2), "principales_gastos_por_categoria": gastos_por_categoria.to_dict()}
    except Exception as e:
        print(f"Error en analisis_descriptivo: {e}")
        return {"error": f"Error en análisis descriptivo: {str(e)}"}

def analisis_predictivo(df):
    try:
        df_pred = df.copy()
        df_pred = df_pred.dropna(subset=['fecha', 'monto', 'tipo', 'categoria'])
        if df_pred.empty: return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "Datos insuficientes post-limpieza."}
        df_pred['mes'] = df_pred['fecha'].dt.month
        df_pred['dia_semana'] = df_pred['fecha'].dt.dayofweek
        df_gastos = df_pred.loc[df_pred['tipo'] == 'gasto'].copy()
        if df_gastos.empty or len(df_gastos) < 5: return {"proyeccion_gastos_proximo_mes": 0, "mensaje": f"Insuficientes registros de gastos ({len(df_gastos)} < 5) para predicción."}
        features = ['mes', 'dia_semana', 'categoria']
        target = 'monto'
        if not all(f in df_gastos.columns for f in features): return {"error": "Faltan columnas ('mes', 'dia_semana', 'categoria') para predicción."}
        X = df_gastos[features]
        y = df_gastos[target]
        categorical_features = ['categoria']
        if X[categorical_features].nunique().iloc[0] <= 1:
             print("Advertencia: Solo una o ninguna categoría de gasto distinta.")
             return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "Se requiere más de una categoría de gasto para la predicción."}
        preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)], remainder='passthrough')
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        model.fit(X, y)
        mes_maximo_datos = df_pred['fecha'].dt.month.max()
        proximo_mes = (mes_maximo_datos % 12) + 1
        categorias_unicas = df_gastos['categoria'].unique()
        dias_semana_unicos = range(7)
        X_pred_data = []
        for cat in categorias_unicas:
            for dia in dias_semana_unicos: X_pred_data.append([proximo_mes, dia, cat])
        if not X_pred_data: return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "No se generaron datos sintéticos para predicción."}
        X_pred = pd.DataFrame(X_pred_data, columns=['mes', 'dia_semana', 'categoria'])
        X_pred = X_pred[features]
        predicciones = model.predict(X_pred)
        gasto_proyectado_total = np.sum(predicciones[predicciones > 0])
        return { "proyeccion_gastos_proximo_mes": round(gasto_proyectado_total, 2) }
    except Exception as e:
        print(f"ERROR DETALLADO en analisis_predictivo: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error crítico en análisis predictivo: {str(e)}"}


# === FUNCIONES DE IA (usando OpenRouter) ===
# (Sin cambios respecto a la versión anterior completa)
def _llamar_openai(prompt: str, system_message: str, model: str = "openai/gpt-4o", temperature: float = 0.7, max_tokens: int = 800) -> dict:
    if not IA_CLIENT: return {"recomendacion": "Error interno: El servicio de IA no está configurado."}
    try:
        print(f"Llamando a OpenRouter con modelo '{model}'...")
        response = IA_CLIENT.chat.completions.create(model=model, messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
        print("Llamada a OpenRouter exitosa.")
        ia_content = response.choices[0].message.content if response.choices else "No se recibió respuesta de IA."
        return {"recomendacion": ia_content}
    except Exception as e:
        print(f"Error EXCEPCIÓN llamando a IA (OpenRouter): {type(e).__name__} - {e}")
        status_code_str = ""; error_detail = str(e)
        if hasattr(e, 'status_code'):
             status_code_str = f"Error code: {e.status_code} - "
             try:
                error_body = getattr(e, 'body', None); error_message_key = getattr(e, 'message', None)
                if isinstance(error_body, dict) and 'error' in error_body and 'message' in error_body['error']: error_detail = error_body['error']['message']
                elif error_message_key: error_detail = error_message_key # Usar el atributo message si existe
                elif isinstance(error_body, str): error_detail = error_body
             except: pass
        return {"recomendacion": f"Error al contactar la IA (OpenRouter): {status_code_str}{error_detail}"}


def generar_respuesta_ia(descriptivo: dict, predictivo: dict) -> dict:
    if isinstance(descriptivo, dict) and "error" in descriptivo: return {"recomendacion": f"IA no generada: Error en descriptivo - {descriptivo['error']}"}
    if isinstance(predictivo, dict) and "error" in predictivo:
         print(f"Advertencia IA: Falló predictivo ({predictivo['error']}), IA con datos descriptivos.")
         predictivo = {"proyeccion_gastos_proximo_mes": 0}
    gastos_str = "\n".join([f"  - {cat}: ${monto:,.2f}" for cat, monto in descriptivo.get('principales_gastos_por_categoria', {}).items()]) or "No se registraron gastos significativos."
    prompt = f"""Eres consultor financiero senior Banorte (riesgos/estrategia). Datos recibidos. **Resumen:** * Ingresos: ${descriptivo.get('total_ingresos', 0):,.2f} * Gastos: ${descriptivo.get('total_gastos', 0):,.2f} * Balance Neto: ${descriptivo.get('balance_neto', 0):,.2f} * Principales Gastos:\n{gastos_str} * Proyección Gastos (Próx Mes): ${predictivo.get('proyeccion_gastos_proximo_mes', 0):,.2f}. **Tarea:** Análisis exhaustivo (riesgos/escenarios), tono formal. Estructura: 1. **Diagnóstico Actual:** (Salud gral). 2. **Riesgos Potenciales (3 Clave):** (Basado en datos; explicar por qué). 3. **Escenarios Posibles (Cualitativo):** Pesimista (si riesgo se materializa), Optimista (con correcciones). 4. **Soluciones/Mitigación:** (1 accionable por riesgo). 5. **Conclusión Profesional:** (Resumen, recomendación resiliencia). **Importante:** Solo datos provistos. Lenguaje claro, profesional, accionable."""
    system_message = "Eres un consultor financiero senior de Banorte, experto en riesgos y estrategia."
    return _llamar_openai(prompt, system_message, model="openai/gpt-4o", max_tokens=800) # Revisa modelo


def generar_plan_meta(meta: str, descriptivo: dict, predictivo: dict) -> dict:
    if isinstance(descriptivo, dict) and "error" in descriptivo: return {"recomendacion": f"Plan no generado: Error en descriptivo - {descriptivo['error']}"}
    if isinstance(predictivo, dict) and "error" in predictivo:
         print(f"Advertencia Meta: Falló predictivo ({predictivo['error']}), plan con datos descriptivos.")
         predictivo = {"proyeccion_gastos_proximo_mes": 0}
    gastos_str = "\n".join([f"  - {cat}: ${monto:,.2f}" for cat, monto in descriptivo.get('principales_gastos_por_categoria', {}).items()]) or "No se registraron gastos."
    prompt = f"""Eres coach financiero Banorte (metas). Usuario necesita ayuda. **Meta:** "{meta}". **Situación:** * Ingresos: ${descriptivo.get('total_ingresos', 0):,.2f} * Gastos: ${descriptivo.get('total_gastos', 0):,.2f} * Cap. Ahorro: ${descriptivo.get('balance_neto', 0):,.2f} * Gastos Proy.: ${predictivo.get('proyeccion_gastos_proximo_mes', 0):,.2f} * Principales Gastos:\n{gastos_str}. **Tarea:** Plan paso a paso (3-5 pasos). Realista (basado en ahorro; si meta irreal, sugerir ajustes). Específico (mencionar categorías/montos). Accionable (pasos claros). Motivador (nota final). Tono amigable-profesional."""
    system_message = "Eres un coach financiero de Banorte, enfocado en metas."
    return _llamar_openai(prompt, system_message, model="openai/gpt-4", max_tokens=700) # Revisa modelo


# === PROCESADORES PRINCIPALES ===

def procesar_finanzas_personales(file_stream: io.BytesIO) -> dict:
    try:
        df = pd.read_excel(file_stream)
        df_clean = clean_data_personal(df)
        if isinstance(df_clean, dict) and "error" in df_clean: return df_clean
        descriptivo = analisis_descriptivo(df_clean)
        if isinstance(descriptivo, dict) and "error" in descriptivo: return descriptivo
        predictivo = analisis_predictivo(df_clean)
        respuesta_ia = generar_respuesta_ia(descriptivo, predictivo)
        return { "tipo_archivo": "Finanzas Personales", "analisis_descriptivo": descriptivo, "analisis_predictivo": predictivo if not (isinstance(predictivo, dict) and "error" in predictivo) else {"mensaje": predictivo.get("mensaje", "Error en predicción")}, "analisis_ia": respuesta_ia }
    except Exception as e:
        print(f"Excepción INESPERADA en procesar_finanzas_personales: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()
        return {"error": f"Error inesperado procesando archivo personal: {str(e)}"}

def procesar_finanzas_empresa(file_stream: io.BytesIO) -> dict:
    try:
        df = pd.read_excel(file_stream)
        df_clean = clean_data_empresa(df)
        if isinstance(df_clean, dict) and "error" in df_clean: return df_clean
        descriptivo = analisis_descriptivo(df_clean)
        if isinstance(descriptivo, dict) and "error" in descriptivo: return descriptivo
        predictivo = analisis_predictivo(df_clean)
        respuesta_ia = generar_respuesta_ia(descriptivo, predictivo)
        return { "tipo_archivo": "Finanzas de Empresa", "analisis_descriptivo": descriptivo, "analisis_predictivo": predictivo if not (isinstance(predictivo, dict) and "error" in predictivo) else {"mensaje": predictivo.get("mensaje", "Error en predicción")}, "analisis_ia": respuesta_ia }
    except Exception as e:
        print(f"Excepción INESPERADA en procesar_finanzas_empresa: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()
        return {"error": f"Error inesperado procesando archivo de empresa: {str(e)}"}

# === NUEVA FUNCIÓN DE SIMULACIÓN (CORREGIDA) ===

# Usa el modelo Pydantic definido aquí o importado
def ejecutar_simulacion_escenarios(datos_base: dict, meses: int, cambios_escenario: List[CambioSimulacionModel]) -> dict: # Usa el modelo Pydantic
    try:
        periodo_meses_base = 12
        if periodo_meses_base <= 0: periodo_meses_base = 1
        ingreso_mensual_base = datos_base.get('total_ingresos', 0) / periodo_meses_base
        gastos_mensuales_base_cat = { cat: monto / periodo_meses_base for cat, monto in datos_base.get('principales_gastos_por_categoria', {}).items() }
        balance_inicial = 0 # Proyectar cambio neto
        fechas_proyeccion = []
        balance_proyectado = []
        balance_actual = balance_inicial
        fecha_actual = datetime.now()

        for i in range(meses):
            fecha_mes_siguiente = fecha_actual + relativedelta(months=i + 1, day=31)
            fechas_proyeccion.append(fecha_mes_siguiente.strftime('%Y-%m-%d'))
            ingresos_mes = ingreso_mensual_base
            gastos_mes_detalle = gastos_mensuales_base_cat.copy()
            gastos_fijos_generales_adicionales = 0 # Para cambios de gasto sin categoría

            # --- ¡CORRECCIÓN AQUÍ! Accede a atributos con punto ---
            for cambio in cambios_escenario: # cambio es un objeto CambioSimulacionModel
                if cambio.tipo == 'ingreso':
                    if cambio.porcentaje_cambio is not None: ingresos_mes *= (1 + cambio.porcentaje_cambio)
                    if cambio.monto_fijo_cambio is not None: ingresos_mes += cambio.monto_fijo_cambio
                elif cambio.tipo == 'gasto':
                    if cambio.categoria: # Cambio a categoría específica
                        if cambio.categoria in gastos_mes_detalle:
                            if cambio.porcentaje_cambio is not None: gastos_mes_detalle[cambio.categoria] *= (1 + cambio.porcentaje_cambio)
                            if cambio.monto_fijo_cambio is not None: gastos_mes_detalle[cambio.categoria] += cambio.monto_fijo_cambio
                            gastos_mes_detalle[cambio.categoria] = max(0, gastos_mes_detalle[cambio.categoria])
                        elif cambio.monto_fijo_cambio is not None and cambio.monto_fijo_cambio > 0:
                             gastos_mes_detalle[cambio.categoria] = cambio.monto_fijo_cambio # Añade nueva categoría
                    else: # Cambio general a gastos
                         if cambio.porcentaje_cambio is not None:
                              for cat in gastos_mes_detalle: gastos_mes_detalle[cat] *= (1 + cambio.porcentaje_cambio)
                         if cambio.monto_fijo_cambio is not None:
                              gastos_fijos_generales_adicionales += cambio.monto_fijo_cambio # Suma al total después

            gastos_mes_total = sum(gastos_mes_detalle.values()) + gastos_fijos_generales_adicionales
            gastos_mes_total = max(0, gastos_mes_total)
            balance_neto_mes = ingresos_mes - gastos_mes_total
            balance_actual += balance_neto_mes
            balance_proyectado.append(round(balance_actual, 2))

        return {"fechas": fechas_proyeccion, "proyeccion": balance_proyectado}

    except Exception as e:
        print(f"Error en ejecutar_simulacion_escenarios: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error ejecutando la simulación: {str(e)}"}

# --- FIN DEL ARCHIVO app/processor.py ---