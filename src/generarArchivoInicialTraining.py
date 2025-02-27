import pandas as pd
from src.data_loader import load_data, save_to_feather, load_data_tasas
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
from src.data_processing import join

#TODO:  falta parametro de inicio de trining
#TODO:  incluir un parámetro con un texto que indique las columnas adicionales a incluir:  'cd_uneg_cont','cd_sucu','cd_marca','cd_line_vehi'

def generar_archivo_inicial_training(horizonte, historia_ventas, historia_cotizaciones, historia_tasas, historia_disponibles, fechaDeCorte):
    """
    Genera un archivo dfData listo para el entrenamiento y un Excel con los parámetros de entrada.
    """

    fechaDeCorte = pd.to_datetime(fechaDeCorte)

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
    dfVentas = process_data_ventas(dfVentas, dfLdP)

    # Calcular cotizaciones por semana por modelo
    print('va a process_data_cotizaciones...')
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones)

    # Calcular tasas por semana
    print('va a process_data_tasas...')
    dfTasas = process_data_tasas(dfTasas)

    # Calcular disponibles por semana por modelo
    print('va a process_data_disponibles...')
    dfDisponibles = process_data_disponibles(dfDisponibles)

    # Hacer el join de los DataFrames
    print('va a join...')
    dfData = join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles)


    # Procesar las columnas de acuerdo al horizonte y la historia
    for i in range(0, horizonte):
        col_ventas = f'VENTAS_{i + 1}'
        col_cotizaciones = f'COTIZACIONES_{i}'
        col_tasas = f'TASAS_{i}'
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

    print("COLUMNAS EN DFDATA:" , dfData.columns)
    
    # Guardar el archivo en formato Feather
    save_to_feather(dfData, 'Data Original/dfData.feather')

    #TODO:  vamos a ejecutar esto muchas veces.  cada vez va a generar un archivo.
    # paso 1:  lee el dataframe parametrosentrenamiento.
    # paso 2:  mirar cuál es la ultima fila y obtienes el siguiente id (idMaximo)
    # paso 3:  insertar una nueva fila con los parametro y creas un archivo nuevo CON OTRO NOMBRE idMaximo
    
    # Guardar los parámetros en un archivo Excel
    parametros = {
        #TODO:  nombreArchivo:33.xlsx
        #TODO:  fecha y hora de hoy
        'horizonte': [horizonte],
        'historia_ventas': [historia_ventas],
        'historia_cotizaciones': [historia_cotizaciones],
        'historia_tasas': [historia_tasas],
        'historia_disponibles': [historia_disponibles],
        'fechaDeCorte': [fechaDeCorte]
    }
    #TODO: reemplazar 
    dfParametros = pd.DataFrame(parametros)
    dfParametros.to_excel('Data Training/ParametrosEntrenamiento.xlsx', index=False)
    
    print("✅ Archivo dfData.feather y ParametrosEntrenamiento.xlsx generados correctamente.")


