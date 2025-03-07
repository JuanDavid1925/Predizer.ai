import pandas as pd
from src.data_loader import load_data, load_data_tasas
from src.data_processing import process_data_precios, join
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
from src.data_loader import save_to_feather
import ast  # Importar módulo para convertir texto a lista
from datetime import datetime

def generar_archivo_prediccion(semana_prediccion, modelos, id_modelo):
    """
    Genera un archivo dfData listo para realizar predicciones con el modelo entrenado.
    """
    fecha_actual = datetime.now().strftime("%Y-%m-%d")

    # Definir ruta del archivo de parámetros
    ruta_parametros_entrenamiento = 'Data Training/ParametrosEntrenamiento.xlsx'
    
    # Cargar los parámetros de entrenamiento para el modelo
    dfParametrosEntrenamiento = pd.read_excel(ruta_parametros_entrenamiento)
    parametros = dfParametrosEntrenamiento[dfParametrosEntrenamiento['nombreArchivo'] == id_modelo]
    
    if parametros.empty:
        raise ValueError(f"No se encontraron parámetros de entrenamiento para el modelo con ID {id_modelo}.")
    
    historia_ventas = parametros['historia_ventas'].values[0]
    historia_cotizaciones = parametros['historia_cotizaciones'].values[0]
    historia_tasas = parametros['historia_tasas'].values[0]
    historia_disponibles = parametros['historia_disponibles'].values[0]
    horizonte = parametros['horizonte'].values[0]
    columnas_adicionales_entrenamiento = ast.literal_eval(parametros['columnasAdicionales'].values[0])
    
    # Construir las columnas necesarias según el entrenamiento
    columnas_adicionales = []
    columnas_adicionales += [f'VENTAS_{i}' for i in range(horizonte, historia_ventas + 1)]
    columnas_adicionales += [f'COTIZACIONES_{i}' for i in range(horizonte, historia_cotizaciones + 1)]
    columnas_adicionales += [f'TASA_CONSUMO_{i}' for i in range(horizonte, historia_tasas + 1)]
    columnas_adicionales += [f'DISPONIBLES_{i}' for i in range(horizonte, historia_disponibles + 1)]
    
    # Cargar los datos originales
    dfVentas = load_data("Data Original/ventas.feather")
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfCotizaciones = load_data("Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")

    print('va a process_data_precios...')
    dfLdP = process_data_precios(dfLdP)

    # Calcular ventas por semana por modelo
    print('va a process_data_ventas...')
    dfVentas = process_data_ventas(dfVentas, dfLdP, semana_prediccion, fecha_actual,columnas_adicionales_entrenamiento)

    # Calcular cotizaciones por semana por modelo
    print('va a process_data_cotizaciones...')
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones, semana_prediccion, fecha_actual)

    # Calcular tasas por semana
    print('va a process_data_tasas...')
    dfTasas = process_data_tasas(dfTasas, semana_prediccion, fecha_actual)

    # Calcular disponibles por semana por modelo
    print('va a process_data_disponibles...')
    dfDisponibles = process_data_disponibles(dfDisponibles, semana_prediccion, fecha_actual)

    # Hacer el join de los DataFrames
    print('va a join...')
    dfDataPrueba = join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles, semana_prediccion, semana_prediccion) 
    
    # Filtrar modelos disponibles en la semana de predicción
    dfModelosPrediccion = pd.DataFrame({'SEMANA': [semana_prediccion] * len(modelos), 'cd_mode_come': modelos})
    
    dfDataPrueba = dfDataPrueba.merge(dfModelosPrediccion, on=['SEMANA', 'cd_mode_come'], how='right')

    # Seleccionar columnas relevantes para predicción
    columnas_finales = ['SEMANA', 'cd_mode_come'] + columnas_adicionales
    dfDataPrueba = dfDataPrueba[columnas_finales]

    
    print(f"✅ Archivo de PREDICCIONES generado correctamente.")

    #save_to_feather(dfDataPrueba, 'Data Training/dfPruebas_1.feather')
    #dfDataPrueba.to_excel('Data Predictions/prediccion.xlsx', index=False)

    return dfDataPrueba
