import pandas as pd
from src.data_loader import load_data, save_to_feather, load_data_tasas
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
from src.data_processing import join
from datetime import datetime
import os

#TODO:  falta parametro de inicio de trining
#TODO:  incluir un parámetro con un texto que indique las columnas adicionales a incluir:  'cd_uneg_cont','cd_sucu','cd_marca','cd_line_vehi'

def generar_archivo_inicial_training(horizonte, historia_ventas, historia_cotizaciones, historia_tasas, historia_disponibles, fechaDeCorte, fechaInicioTraining, columnasAdicionales):
    """
    Genera un archivo dfData listo para el entrenamiento y un Excel con los parámetros de entrada.
    """

    fechaDeCorte = pd.to_datetime(fechaDeCorte)
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Definir ruta del archivo de parámetros
    ruta_parametros = 'Data Training/ParametrosEntrenamiento.xlsx'

    print("Va a leer parametros de entrenamiento...")
    # Verificar si el archivo de parámetros existe
    if os.path.exists(ruta_parametros):
        dfParametros = pd.read_excel(ruta_parametros)
        idMaximo = dfParametros['nombreArchivo'].max() + 1
    else:
        dfParametros = pd.DataFrame(columns=['nombreArchivo', 'fechaDeCreacion', 'horizonte', 'historia_ventas', 'historia_cotizaciones', 'historia_tasas', 'historia_disponibles', 'fechaInicioTraining','fechaDeCorte', 'columnasAdicionales'])
        idMaximo = 1

    nombre_archivo = f"Data Training/dfData_{idMaximo}.feather"

    # Cargar los datos originales
    print('iniciando...')
    dfVentas = load_data("Data Original/ventas.feather")
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfCotizaciones = load_data("Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")
    

    print('va a process_data_precios...')
    dfLdP = process_data_precios(dfLdP)

    # Calcular ventas por semana por modelo
    print('va a process_data_ventas...')
    dfVentas = process_data_ventas(dfVentas, dfLdP, fechaInicioTraining, columnasAdicionales)

    # Calcular cotizaciones por semana por modelo
    print('va a process_data_cotizaciones...')
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones)

    # Calcular tasas por semana
    print('va a process_data_tasas...')
    dfTasas = process_data_tasas(dfTasas)

    # Calcular disponibles por semana por modelo
    print('va a process_data_disponibles...')
    dfDisponibles = process_data_disponibles(dfDisponibles, fechaInicioTraining)

    # Hacer el join de los DataFrames
    print('va a join...')
    dfData = join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles, fechaInicioTraining, columnasAdicionales)


    # Procesar las columnas de acuerdo al horizonte y la historia
    for i in range(1, horizonte):
        col_ventas = f'VENTAS_{i}'
        col_cotizaciones = f'COTIZACIONES_{i}'
        col_tasas = f'TASA_CONSUMO_{i}'
        col_disponibles = f'DISPONIBLES_{i}'
        
        if col_ventas in dfData.columns:
            dfData.drop(columns=[col_ventas], inplace=True)
        if col_cotizaciones in dfData.columns:
            dfData.drop(columns=[col_cotizaciones], inplace=True)
        if col_tasas in dfData.columns:
            dfData.drop(columns=[col_tasas], inplace=True)
        if col_disponibles in dfData.columns:
            dfData.drop(columns=[col_disponibles], inplace=True)

    
    columnas_existentes = dfData.columns.tolist()

    # Mantener solo hasta la historia definida, verificando que las columnas existen
    cols_historia_ventas = [f'VENTAS_{i}' for i in range(historia_ventas + 1) if f'VENTAS_{i}' in columnas_existentes]
    cols_historia_cotizaciones = [f'COTIZACIONES_{i}' for i in range(historia_cotizaciones + 1) if f'COTIZACIONES_{i}' in columnas_existentes]
    cols_historia_tasas = [f'TASAS_{i}' for i in range(historia_tasas + 1) if f'TASAS_{i}' in columnas_existentes]
    cols_historia_disponibles = [f'DISPONIBLES_{i}' for i in range(historia_disponibles + 1) if f'DISPONIBLES_{i}' in columnas_existentes]
    
    columnas_finales = ['SEMANA', 'cd_mode_come'] + cols_historia_ventas + cols_historia_cotizaciones + cols_historia_tasas + cols_historia_disponibles
    dfData = dfData[columnas_finales]

    print("COLUMNAS EN DFDATA DESPUES DE ELIMINAR:" , dfData.columns)
    
    # Guardar el archivo en formato Feather
    save_to_feather(dfData, nombre_archivo)

    # Agregar nueva fila de parámetros
    nueva_fila = pd.DataFrame({
        'nombreArchivo': [idMaximo],
        'fechaDecreacion': [fecha_actual],
        'horizonte': [horizonte],
        'historia_ventas': [historia_ventas],
        'historia_cotizaciones': [historia_cotizaciones],
        'historia_tasas': [historia_tasas],
        'historia_disponibles': [historia_disponibles],
        'fechaInicioTraining': [fechaInicioTraining],
        'fechaDeCorte': [fechaDeCorte],
        'columnasAdicionales' : [columnasAdicionales]
    })

    dfParametros = pd.concat([dfParametros, nueva_fila], ignore_index=True)
    dfParametros.to_excel(ruta_parametros, index=False)
    
    print(f"✅ Archivo {nombre_archivo} generado y registrado en {ruta_parametros} correctamente.")


