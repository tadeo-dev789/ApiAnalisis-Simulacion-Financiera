import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import io
from typing import Dict

class DataProcessor:
    """
    Versión SÚPER-OPTIMIZADA (pivot_table)
    + Lógica de FILTRADO DE USUARIO (Corregida)
    + Cálculo de promedios para IA
    + Cálculo de REPORTE COMPLETO por categoría
    """
    
    def __init__(self, file_content: bytes, filename: str):
        print("Iniciando nuevo procesador...")
        
        # 1. Cargar datos
        try:
            if filename.endswith('.csv'):
                self.df_raw = pd.read_csv(io.BytesIO(file_content))
            elif filename.endswith('.xlsx'):
                self.df_raw = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError("Formato de archivo no soportado. Sube .csv o .xlsx")
        except Exception as e:
            raise ValueError(f"El archivo no es válido o está corrupto: {e}")

        # --- Lógica de FILTRADO DE USUARIO ---
        if 'id_usuario' in self.df_raw.columns:
            usuarios_limpios = self.df_raw['id_usuario'].dropna()
            if usuarios_limpios.empty:
                self.df = self.df_raw.copy()
            else:
                try:
                    top_user_id = usuarios_limpios.mode()[0]
                    self.df = self.df_raw[self.df_raw['id_usuario'] == top_user_id].copy()
                except Exception as e:
                    self.df = self.df_raw.copy()
        else:
            self.df = self.df_raw.copy()
        # --- FIN DE LA LÓGICA DE FILTRADO ---

        # Inicializar variables
        self.ingreso_promedio_calculado = 0
        self.gasto_promedio_calculado = 0
        self.resumen_gastos_por_categoria = {} 
        self.gasto_actual_restaurantes = 500 # Default por si falla
            
        # 2. Ejecutar procesamiento y entrenamiento
        self._process_data_super_optimizado() 
        self._train_model()
        print("Procesador listo (Optimizado + Filtrado Correcto).")

    def _process_data_super_optimizado(self):
        print("Procesando datos (modo SÚPER rápido)...")
        self.df['fecha'] = pd.to_datetime(self.df['fecha']) 
        self.df['monto'] = pd.to_numeric(self.df['monto'], errors='coerce')
        self.df = self.df.dropna(subset=['monto', 'tipo'])
        
        self.df['mes_periodo'] = pd.PeriodIndex(self.df['fecha'], freq='M')
        
        df_pivot = self.df.pivot_table(
            index='mes_periodo', 
            columns='tipo', 
            values='monto', 
            aggfunc='sum'
        ).fillna(0)
        
        if 'ingreso' not in df_pivot.columns: df_pivot['ingreso'] = 0
        if 'gasto' not in df_pivot.columns: df_pivot['gasto'] = 0

        df_pivot['ahorro_mensual'] = df_pivot['ingreso'] - df_pivot['gasto']

        if not df_pivot.empty:
            self.ingreso_promedio_calculado = df_pivot['ingreso'].mean()
            self.gasto_promedio_calculado = df_pivot['gasto'].mean()
        
        self.df_mensual_historico = df_pivot.reset_index()
        self.df_mensual_historico['fecha'] = self.df_mensual_historico['mes_periodo'].dt.to_timestamp(how='end')
        self.df_mensual_historico['ahorro_acumulado'] = self.df_mensual_historico['ahorro_mensual'].cumsum()
        self.df_mensual_historico['mes_num'] = np.arange(len(self.df_mensual_historico))
        self.df_mensual_historico = self.df_mensual_historico[['fecha', 'ahorro_mensual', 'ahorro_acumulado', 'mes_num']]

        # --- Calcular REPORTE COMPLETO ---
        df_gastos = self.df[self.df['tipo'] == 'gasto']
        try:
            num_meses = len(self.df_mensual_historico)
            if num_meses == 0: num_meses = 1

            if not df_gastos.empty:
                gastos_totales_por_categoria = df_gastos.groupby('categoria')['monto'].sum()
                gastos_promedio_por_categoria = (gastos_totales_por_categoria / num_meses).round(2)
                self.resumen_gastos_por_categoria = gastos_promedio_por_categoria.to_dict()
                print(f"Reporte de gastos calculado: {self.resumen_gastos_por_categoria}")
            
            # (Seguimos guardando esto por si quieres mostrarlo como "destacado")
            if 'Restaurantes' in self.resumen_gastos_por_categoria:
                 self.gasto_actual_restaurantes = self.resumen_gastos_por_categoria['Restaurantes']

        except Exception:
            self.gasto_actual_restaurantes = 500 # Valor por defecto
        
        print(f"Datos procesados. Gasto promedio 'Restaurantes' (destacado): {self.gasto_actual_restaurantes}")

    
    def _train_model(self):
        print("Entrenando modelo...")
        X = self.df_mensual_historico[['mes_num']]
        y = self.df_mensual_historico['ahorro_acumulado']
        
        if X.empty or y.empty or len(X) < 2:
             print("Advertencia: No hay suficientes datos (se necesitan 2+) para entrenar un modelo.")
             self.modelo_base = None
             self.proyeccion_base = np.zeros(24)
             fecha_inicio_proyeccion = pd.Timestamp.now()
             if not self.df_mensual_historico['fecha'].empty:
                fecha_inicio_proyeccion = self.df_mensual_historico['fecha'].max()
             self.fechas_futuras = pd.date_range(start=fecha_inicio_proyeccion, periods=25, freq='ME')[1:].strftime('%Y-%m-%d').tolist()
             return

        self.modelo_base = LinearRegression()
        self.modelo_base.fit(X, y)
        
        ultimo_mes_num = self.df_mensual_historico['mes_num'].max()
        meses_futuros_num = np.arange(ultimo_mes_num + 1, ultimo_mes_num + 25).reshape(-1, 1) # 24 meses
        self.proyeccion_base = self.modelo_base.predict(meses_futuros_num)
        
        fecha_inicio_proyeccion = self.df_mensual_historico['fecha'].max()
        self.fechas_futuras = pd.date_range(start=fecha_inicio_proyeccion, periods=25, freq='ME')[1:].strftime('%Y-%m-%d').tolist()
        print("Modelo entrenado.")


    # --- *** ¡FUNCIÓN DE SIMULACIÓN ACTUALIZADA! *** ---
    
    def get_simulation_full_report(self, gastos_simulados: Dict[str, float]) -> dict:
        """
        Endpoint RÁPIDO: Calcula la simulación basado en el REPORTE COMPLETO.
        """
        
        # 1. Obtener los gastos originales (el reporte que ya calculamos)
        gastos_originales_dict = self.resumen_gastos_por_categoria
        
        # 2. Calcular el ahorro total
        
        # Sumamos los gastos originales
        total_original = sum(gastos_originales_dict.values())
        
        # Sumamos los nuevos gastos simulados que nos envió el frontend
        # (Nos aseguramos que solo sume categorías que existían originalmente)
        total_simulado = 0
        for categoria, monto_original in gastos_originales_dict.items():
            # Si el frontend envía una nueva categoría, la usa. Si no, usa la original.
            total_simulado += gastos_simulados.get(categoria, monto_original)

        # El ahorro es la diferencia
        ahorro_extra_mensual = total_original - total_simulado
        print(f"Ahorro mensual simulado total: {ahorro_extra_mensual}")
        
        # 3. Aplicar este ahorro a la proyección (lógica del modelo)
        if self.modelo_base is None:
            ahorro_extra_acumulado = ahorro_extra_mensual * np.arange(1, 25)
            proyeccion_simulada = ahorro_extra_acumulado # Proyección desde cero
        else:
            ahorro_extra_acumulado = ahorro_extra_mensual * np.arange(1, 25) # 24 meses
            # Aplicamos el ahorro extra a la predicción base
            proyeccion_simulada = self.proyeccion_base + ahorro_extra_acumulado
        
        return {
            "fechas": self.fechas_futuras,
            "proyeccion": proyeccion_simulada.tolist()
        }
    
    
    # --- *** ¡FUNCIÓN DE DATOS INICIALES ACTUALIZADA! *** ---

    def get_initial_data(self) -> dict:
        """
        Devuelve los datos iniciales que el frontend necesita.
        ¡AHORA INCLUYE EL REPORTE COMPLETO!
        """
        return {
            ### ¡CAMBIO IMPORTANTE! ###
            # "gasto_actual_restaurantes": self.gasto_actual_restaurantes, # Ya no se usa
            
            # Envía el reporte completo para que el frontend pueda crear todos los sliders
            "reporte_gastos_actual": self.resumen_gastos_por_categoria,
            
            "proyeccion_base": self.proyeccion_base.tolist(),
            "fechas": self.fechas_futuras,
            "datos_historicos": { 
                "fechas": self.df_mensual_historico['fecha'].dt.strftime('%Y-%m-%d').tolist(),
                "ahorro_acumulado": self.df_mensual_historico['ahorro_acumulado'].tolist()
            }
        }