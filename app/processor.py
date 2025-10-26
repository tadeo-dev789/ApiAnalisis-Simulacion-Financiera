# app/processor.py

import pandas as pd
from openai import OpenAI # Se sigue usando la librería de OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import io

# --- LÓGICA DE CARGA DE .ENV ---
from dotenv import load_dotenv
from pathlib import Path

# Construye la ruta al archivo .env relativo a este archivo
env_path = Path(__file__).parent.parent / ".env"
# Carga el .env desde esa ruta específica
load_dotenv(dotenv_path=env_path)
# ---------------------------------

# --- ¡CAMBIO 1: Usa la clave de OpenRouter! ---
# Carga la API Key de OpenRouter desde el entorno
API_KEY_CARGADA = os.environ.get("OPENROUTER_API_KEY") # <-- CAMBIAR NOMBRE EN .env

# (Puedes mantener o quitar el print de depuración)
print("="*50)
print(f"DEBUG: Clave de API cargada: {API_KEY_CARGADA}")
print("="*50)


# --- ¡CAMBIO 2: Configura el cliente para OpenRouter! ---
IA_CLIENT = None # Inicializa como None
if not API_KEY_CARGADA:
    print("ERROR FATAL: La variable de entorno OPENROUTER_API_KEY no se encontró o está vacía.")
    # Considera lanzar una excepción aquí
    # raise ValueError("Falta la clave de API de OpenRouter en el entorno.")
else:
    try:
        # Configura el cliente para usar OpenRouter
        IA_CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1", # <-- URL de OpenRouter
            api_key=API_KEY_CARGADA,
            default_headers={ # <-- Cabeceras requeridas por OpenRouter
                "HTTP-Referer": "http://localhost", # O tu URL de frontend
                "X-Title": "Banorte Analisis Financiero", # Nombre de tu app
            },
        )
        print("Cliente OpenAI inicializado CORRECTAMENTE, apuntando a OpenRouter.")
    except Exception as e:
        print(f"ERROR: Falló la inicialización del cliente OpenAI para OpenRouter: {e}")
        IA_CLIENT = None


# --- LÓGICA DE LIMPIEZA PARA FINANZAS PERSONALES ---
# (Sin cambios aquí)
def clean_data_personal(df):
    try:
        column_mapping = {
            'id_usuario': 'id', 'monto': 'monto', 'tipo': 'tipo',
            'categoria': 'categoria', 'descripcion': 'descripcion', 'fecha': 'fecha'
        }
        columnas_requeridas = ['id_usuario', 'monto', 'tipo', 'fecha']
        if not all(col in df.columns for col in columnas_requeridas):
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            return {"error": f"Faltan columnas requeridas en archivo personal: {', '.join(faltantes)}"}

        df = df.rename(columns=column_mapping)
        df = df[[col for col in column_mapping.values() if col in df.columns]]
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['monto'] = pd.to_numeric(df['monto'], errors='coerce').fillna(0)
        df = df.dropna(subset=['fecha', 'monto', 'tipo'])
        df['categoria'] = df['categoria'].astype(str).fillna('Desconocida')
        df['descripcion'] = df['descripcion'].astype(str).fillna('Sin descripción')
        df['tipo'] = df['tipo'].astype(str).str.lower()
        return df
    except Exception as e:
        print(f"Error en clean_data_personal: {e}")
        return {"error": f"Error limpiando datos personales: {str(e)}"}

# --- LÓGICA DE LIMPIEZA PARA FINANZAS DE EMPRESA ---
# (Sin cambios aquí)
def clean_data_empresa(df):
    try:
        column_mapping = {
            'empresa_id': 'id', 'fecha': 'fecha', 'tipo': 'tipo',
            'concepto': 'descripcion', 'categoria': 'categoria', 'monto': 'monto'
        }
        columnas_requeridas = ['empresa_id', 'fecha', 'tipo', 'monto']
        if not all(col in df.columns for col in columnas_requeridas):
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            return {"error": f"Faltan columnas requeridas en archivo de empresa: {', '.join(faltantes)}"}

        columnas_presentes = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=columnas_presentes)
        columnas_finales = [v for v in column_mapping.values() if v in df.columns]
        df = df[columnas_finales]
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['monto'] = pd.to_numeric(df['monto'], errors='coerce').fillna(0)
        df = df.dropna(subset=['fecha', 'monto', 'tipo'])
        df['categoria'] = df['categoria'].astype(str).fillna('Desconocida')
        df['descripcion'] = df['descripcion'].astype(str).fillna('Sin descripción')
        df['tipo'] = df['tipo'].astype(str).str.lower()
        return df
    except Exception as e:
        print(f"Error en clean_data_empresa: {e}")
        return {"error": f"Error limpiando datos de empresa: {str(e)}"}


# --- FUNCIONES DE ANÁLISIS GENÉRICAS ---
# (Sin cambios aquí)
def analisis_descriptivo(df):
    try:
        df['tipo'] = df['tipo'].astype(str)
        total_ingresos = df[df['tipo'] == 'ingreso']['monto'].sum()
        total_gastos = df[df['tipo'] == 'gasto']['monto'].sum()
        balance = total_ingresos - total_gastos
        gastos_por_categoria = df[df['tipo'] == 'gasto'].groupby('categoria')['monto'].sum().sort_values(ascending=False).round(2)
        return {
            "total_ingresos": round(total_ingresos, 2),
            "total_gastos": round(total_gastos, 2),
            "balance_neto": round(balance, 2),
            "principales_gastos_por_categoria": gastos_por_categoria.to_dict()
        }
    except Exception as e:
        print(f"Error en analisis_descriptivo: {e}")
        return {"error": f"Error en análisis descriptivo: {str(e)}"}

def analisis_predictivo(df):
    try:
        df_pred = df.copy()
        df_pred = df_pred.dropna(subset=['fecha', 'monto', 'tipo', 'categoria'])
        if df_pred.empty:
             return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "No hay datos suficientes después de limpiar nulos."}

        df_pred['mes'] = df_pred['fecha'].dt.month
        df_pred['dia_semana'] = df_pred['fecha'].dt.dayofweek
        df_gastos = df_pred[df_pred['tipo'] == 'gasto'].copy()

        if df_gastos.empty or len(df_gastos) < 5:
            return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "Datos insuficientes de gastos (<5) para predicción fiable."}

        features = ['mes', 'dia_semana', 'categoria']
        target = 'monto'
        X = df_gastos[features]
        y = df_gastos[target]

        categorical_features = ['categoria']
        if X[categorical_features].nunique().iloc[0] <= 1:
             print("Advertencia: Solo se encontró una o ninguna categoría de gasto distinta.")
             # Podríamos devolver un mensaje o intentar predecir sin categoría
             return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "Se requiere más de una categoría de gasto para la predicción."}


        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
            remainder='passthrough'
            )

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])
        model.fit(X, y)

        proximo_mes = (df_pred['mes'].max() % 12) + 1
        categorias_unicas = df_gastos['categoria'].unique()
        dias_semana_unicos = range(7)

        X_pred_data = []
        for cat in categorias_unicas:
            for dia in dias_semana_unicos:
                 X_pred_data.append([proximo_mes, dia, cat])

        if not X_pred_data:
             return {"proyeccion_gastos_proximo_mes": 0, "mensaje": "No se pudieron generar datos sintéticos para predicción."}

        X_pred = pd.DataFrame(X_pred_data, columns=['mes', 'dia_semana', 'categoria'])
        X_pred = X_pred[features]

        predicciones = model.predict(X_pred)
        gasto_proyectado_total = np.sum(predicciones[predicciones > 0])

        return {
            "proyeccion_gastos_proximo_mes": round(gasto_proyectado_total, 2)
        }
    except Exception as e:
        print(f"Error en analisis_predictivo: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error en análisis predictivo: {str(e)}"}


# --- ¡CAMBIO 3: Especifica el modelo de OpenRouter en las llamadas a IA! ---
def _llamar_openai(prompt, system_message, model="openai/gpt-4", temperature=0.7, max_tokens=500): # <-- Modelo de OpenRouter
    """Función helper para llamar a la IA (configurada para OpenRouter) y manejar errores."""
    if not IA_CLIENT:
        print("ERROR: Cliente OpenAI (IA_CLIENT) no está inicializado.")
        return {"error": "El cliente IA no está configurado correctamente. Verifica la API Key de OpenRouter."}
    try:
        # Usa el modelo específico de OpenRouter (ej. 'openai/gpt-4', 'google/gemini-pro', etc.)
        print(f"Llamando a OpenRouter con modelo {model}...")
        response = IA_CLIENT.chat.completions.create(
            model=model, # <-- Asegúrate que este modelo esté disponible en tu cuenta OpenRouter
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Llamada a OpenRouter exitosa.")
        ia_content = response.choices[0].message.content if response.choices else "No se recibió respuesta de IA."
        return {"recomendacion": ia_content} # Cambiado 'analisis_ia' a 'recomendacion' para coincidir con tu error

    except Exception as e:
        print(f"Error EXCEPCIÓN llamando a IA (OpenRouter): {type(e).__name__} - {e}")
        error_message = f"Error al contactar la IA (OpenRouter): {str(e)}"
        status_code_str = ""
        # Intenta extraer el status code si es un error de API
        if hasattr(e, 'status_code'):
             status_code_str = f"Error code: {e.status_code} - "
             try:
                # Intenta parsear el cuerpo del error si es JSON
                error_body = getattr(e, 'body', None)
                if error_body:
                    import json
                    error_json = json.loads(error_body)
                    error_message = f"Error al contactar la IA (OpenRouter): {status_code_str}{str(error_json)}"
                else:
                    # Intenta obtener el mensaje del error si no hay body
                    error_message = f"Error al contactar la IA (OpenRouter): {status_code_str}{getattr(e, 'message', str(e))}"

             except: # Si no se puede parsear, usa el error genérico
                 error_message = f"Error al contactar la IA (OpenRouter): {status_code_str}{str(e)}"


        # Devuelve el error como parte del JSON con la clave 'recomendacion'
        return {"recomendacion": error_message}

def generar_respuesta_ia(descriptivo, predictivo):
    # Verifica si hubo errores en los análisis previos antes de llamar a la IA
    if isinstance(descriptivo, dict) and "error" in descriptivo:
         return {"recomendacion": f"No se pudo generar análisis IA: Error en análisis descriptivo - {descriptivo['error']}"} # Clave 'recomendacion'
    if isinstance(predictivo, dict) and "error" in predictivo:
         print(f"Advertencia: Falló el análisis predictivo ({predictivo['error']}), se generará IA solo con datos descriptivos.")
         predictivo = {"proyeccion_gastos_proximo_mes": 0} # Valor por defecto

    # Formatea los gastos por categoría para el prompt
    gastos_str = "\n".join([f"  - {cat}: ${monto:,.2f}" for cat, monto in descriptivo.get('principales_gastos_por_categoria', {}).items()])
    if not gastos_str:
        gastos_str = "No se registraron gastos significativos."

    # --- NUEVO PROMPT ---
    prompt = f"""
    Eres un consultor financiero senior de Banorte, especializado en análisis de riesgos y planificación estratégica. Has recibido los datos financieros de un cliente.

    **Resumen Financiero Clave:**
    * **Ingresos Totales:** ${descriptivo.get('total_ingresos', 0):,.2f}
    * **Gastos Totales:** ${descriptivo.get('total_gastos', 0):,.2f}
    * **Balance Neto (Flujo de Efectivo):** ${descriptivo.get('balance_neto', 0):,.2f}
    * **Principales Categorías de Gasto:**
{gastos_str}
    * **Proyección Estimada de Gastos (Próximo Mes):** ${predictivo.get('proyeccion_gastos_proximo_mes', 0):,.2f}

    **Tu Tarea:** Realiza un análisis exhaustivo y profesional enfocado en la **anticipación de riesgos** y la **planificación de escenarios**, manteniendo un tono formal y directo. Estructura tu respuesta de la siguiente manera:

    1.  **Diagnóstico Financiero Actual:**
        * Resume la salud financiera general basada en los datos (ej. solvencia, liquidez aparente, estructura de gastos).

    2.  **Identificación Detallada de Riesgos Potenciales:**
        * Basado en la estructura de gastos, el balance neto y la proyección, identifica **3 riesgos financieros clave** a los que el cliente podría enfrentarse (ej. riesgo de liquidez si los ingresos disminuyen, riesgo de concentración de gastos, riesgo de no alcanzar metas de ahorro si los gastos proyectados se materializan).
        * Para cada riesgo, explica *por qué* es un riesgo según los datos presentados.

    3.  **Escenarios Posibles (Simulación Cualitativa):**
        * Describe brevemente **dos escenarios futuros** basados en los riesgos identificados:
            * **Escenario Pesimista:** ¿Qué podría pasar si uno de los riesgos principales se materializa (ej. aumento inesperado en 'costos', caída de 'ventas')? ¿Cuál sería el impacto probable en el balance neto?
            * **Escenario Optimista:** ¿Qué sucedería si se implementan acciones correctivas? ¿Cómo podría mejorar el balance neto o reducirse la exposición al riesgo?

    4.  **Soluciones y Estrategias de Mitigación:**
        * Para cada riesgo identificado en el punto 2, proporciona **una solución o estrategia concreta y accionable** que el cliente puede implementar para mitigarlo o prevenirlo.
        * Sé específico (ej. "Implementar revisión trimestral de gastos en la categoría 'X'", "Diversificar fuentes de ingreso explorando el mercado Y", "Establecer fondo de emergencia equivalente a Z meses de gastos").

    5.  **Conclusión Profesional:**
        * Cierra con un breve resumen y una recomendación general enfocada en la resiliencia financiera.

    **Importante:** Basa tu análisis *únicamente* en los datos proporcionados. Evita suposiciones no fundamentadas. Mantén un lenguaje claro, profesional y orientado a la acción.
    """
    # --- FIN DEL NUEVO PROMPT ---

    system_message = "Eres un consultor financiero senior de Banorte, experto en riesgos y estrategia."
    # Asegúrate de usar un modelo capaz y disponible en OpenRouter, ej. "openai/gpt-4o" o similar
    # Aumentamos ligeramente los tokens para la respuesta detallada
    return _llamar_openai(prompt, system_message, model="openai/gpt-4o", max_tokens=800) # Ajusta el modelo si usas otro

def generar_plan_meta(meta: str, descriptivo: dict, predictivo: dict):
    # (Sin cambios en la lógica del prompt, solo en la llamada a _llamar_openai)
    if isinstance(descriptivo, dict) and "error" in descriptivo:
         return {"recomendacion": f"No se pudo generar plan de meta: Error en análisis descriptivo - {descriptivo['error']}"} # Clave 'recomendacion'
    if isinstance(predictivo, dict) and "error" in predictivo:
         print(f"Advertencia: Falló el análisis predictivo ({predictivo['error']}), se generará plan de meta solo con datos descriptivos.")
         predictivo = {"proyeccion_gastos_proximo_mes": 0}

    gastos_str = "\n".join([f"  - {cat}: ${monto:,.2f}" for cat, monto in descriptivo.get('principales_gastos_por_categoria', {}).items()])
    if not gastos_str: gastos_str = "No se registraron gastos."

    prompt = f"""
    Eres un coach financiero de metas. Usuario necesita ayuda.
    Meta: "{meta}"
    Situación Actual:
    - Ingresos: ${descriptivo.get('total_ingresos', 0):,.2f}
    - Gastos: ${descriptivo.get('total_gastos', 0):,.2f}
    - Capacidad Ahorro: ${descriptivo.get('balance_neto', 0):,.2f}
    - Gastos Proyectados Mes Siguiente: ${predictivo.get('proyeccion_gastos_proximo_mes', 0):,.2f}
    - Principales Gastos:\n{gastos_str}

    Genera plan paso a paso (3-5 pasos) realista, específico (menciona categorías y montos), accionable y motivador. Si la meta es irreal, sugiere ajustes.
    """
    system_message = "Eres un coach financiero de Banorte, enfocado en metas."
    # Usa un modelo disponible en OpenRouter y da más tokens
    return _llamar_openai(prompt, system_message, model="openai/gpt-4", max_tokens=700)


# --- PROCESADOR 1: FINANZAS PERSONALES ---
def procesar_finanzas_personales(file_stream: io.BytesIO):
    try:
        df = pd.read_excel(file_stream)
        df_clean = clean_data_personal(df)
        if isinstance(df_clean, dict) and "error" in df_clean: return df_clean

        descriptivo = analisis_descriptivo(df_clean)
        if isinstance(descriptivo, dict) and "error" in descriptivo: return descriptivo

        predictivo = analisis_predictivo(df_clean)
        # No retorna error si falla predictivo, pero pasa info a IA

        respuesta_ia = generar_respuesta_ia(descriptivo, predictivo)
        # El error de IA ya se maneja dentro y se devuelve en 'recomendacion'

        return {
            "tipo_archivo": "Finanzas Personales",
            "analisis_descriptivo": descriptivo,
            "analisis_predictivo": predictivo if not (isinstance(predictivo, dict) and "error" in predictivo) else {"mensaje": predictivo.get("mensaje", "Error en predicción")},
            "analisis_ia": respuesta_ia # Devuelve el resultado o el error de IA con clave 'recomendacion'
        }
    except Exception as e:
        print(f"Excepción INESPERADA en procesar_finanzas_personales: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error inesperado procesando archivo personal: {str(e)}"}


# --- PROCESADOR 2: FINANZAS DE EMPRESA ---
def procesar_finanzas_empresa(file_stream: io.BytesIO):
    try:
        df = pd.read_excel(file_stream)
        df_clean = clean_data_empresa(df)
        if isinstance(df_clean, dict) and "error" in df_clean: return df_clean

        descriptivo = analisis_descriptivo(df_clean)
        if isinstance(descriptivo, dict) and "error" in descriptivo: return descriptivo

        predictivo = analisis_predictivo(df_clean)
        # No retorna error si falla predictivo

        # Llamamos a la función genérica (podrías crear una específica si necesitas otro prompt)
        respuesta_ia = generar_respuesta_ia(descriptivo, predictivo)

        return {
            "tipo_archivo": "Finanzas de Empresa",
            "analisis_descriptivo": descriptivo,
            "analisis_predictivo": predictivo if not (isinstance(predictivo, dict) and "error" in predictivo) else {"mensaje": predictivo.get("mensaje", "Error en predicción")},
            "analisis_ia": respuesta_ia # Devuelve el resultado o el error de IA con clave 'recomendacion'
        }
    except Exception as e:
        print(f"Excepción INESPERADA en procesar_finanzas_empresa: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error inesperado procesando archivo de empresa: {str(e)}"}