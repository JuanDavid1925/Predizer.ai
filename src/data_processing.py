import datetime
import pandas as pd
import pandasql as psql
from src.data_loader import save_to_feather

def get_previous_monday(date):
    """Obtiene el lunes anterior a una fecha dada."""
    return date - datetime.timedelta(days=date.weekday())

def process_data_ventas(dfVentas, dfLdP):
    columnas = ['fe_pedi_cli','cd_mode_come','cd_cia','cd_uneg_cont','cd_sucu','cd_marca','cd_line_vehi']
    dfVentas = dfVentas[columnas]
    # COMO VAMOS A AGRUPAR POR SEMANA, OBTENEMOS EL LUNES ANTERIOR A CADA FECHA DE PEDIDO
    dfVentas['SEMANA'] = dfVentas['fe_pedi_cli'].apply(get_previous_monday)

    # Y BORRAMOS LA FECHA DE PEDIDO PUES TRABAJAREMOS CON DATOS SEMANALES
    dfVentas = dfVentas.drop(columns=['fe_pedi_cli'])

    # AGRUPAMIENTO POR SEMANA:  obtenemos las ventas por SEMANA y MODELO (cd_mode_come)
    dfVentas = dfVentas.groupby(['SEMANA', 'cd_mode_come','cd_cia','cd_uneg_cont','cd_marca','cd_line_vehi']).agg(VENTAS=('cd_mode_come', 'count')).reset_index()
    dfVentas = dfVentas.loc[dfVentas['SEMANA']>='2015-01-01'].copy()
    dfVentas = dfVentas.sort_values(by=['SEMANA', 'cd_mode_come'])
    dfVentas = dfVentas.reset_index(drop=True)

    # obtenemos las semanas desde el 2015
    distintas_semanas = dfVentas['SEMANA'].unique()

    # para cada semana s


    listaAdicionarFilas = []

    for s in distintas_semanas:
    #   # obtenemos la lista de precios anterior más cercana a s y de esta obtenemos los modelos de dicha lista
    #   # al convertirla a un set, no deberían haber modelos repetidos
        modelosEnListaDePrecios = set(fechaMasCercana(s, dfLdP))

    #   # obtenemos los modelos que SI se vendieron en dicha semana s
    #   # también es un set y por tanto no deben haber modelos repetidos
        modelosEnVentas = set(dfVentas[dfVentas['SEMANA'] == s]['cd_mode_come'].tolist())

    #   # obtenemos los modelos de la lista que no se vendieron.  
        modelosACrear = modelosEnListaDePrecios - modelosEnVentas

    #     # para cada modelo encontrado, creamos una fila en df para esa semana s, ese modelo y con valor 0
        for modelo in modelosACrear:

            fila = (s, modelo, 0)
            listaAdicionarFilas.append(fila)

    # # adicionamos todas las filas que se encontraron
    df = pd.DataFrame(listaAdicionarFilas, columns=['SEMANA','cd_mode_come','VENTAS'])
    dfVentas = pd.concat([dfVentas, df], ignore_index=True)
    
    save_to_feather(dfVentas, './Data Original/dfVentas2.feather')

    return dfVentas


def process_data_precios(dfLdP):
    dfLdP['fecha'] = dfLdP['fe_grab_list'].dt.date
    dfLdP['fecha'] = pd.to_datetime(dfLdP['fecha'])

    # dfLdP nos queda con fecha y modelo
    dfLdP = dfLdP[['fecha','cd_mode_come']].copy()

    return dfLdP


# funcion, que dada una SEMANA retorna de dfLdP los modelos de la fecha más cercana menor a ella
def fechaMasCercana(SEMANA, dfLdP):
    dfTemp = dfLdP.loc[dfLdP['fecha']<SEMANA]  # aqui tenemos las filas en dfLdP menores a SEMANA

    if not dfTemp.empty:
        fila_mas_cercana = dfTemp.loc[dfTemp['fecha'].idxmax()]  # aquí queda la última fila

        # print('fila mas cercana: ', fila_mas_cercana)
        
        # ahora a obtener la lista de modelos 
        modelos = dfLdP[dfLdP['fecha'] == fila_mas_cercana['fecha']]['cd_mode_come'].tolist()
        return modelos
    else:
        return []
    

def process_data_cotizaciones(dfCotizaciones):
    # dejamos solo las columnas que nos interesan
    # columnas = ['fe_grab_pros','cd_cia','cd_uneg_cont','cd_sucu','cd_cope','cd_marca','cd_line_vehi','cd_mode_come']
    columnas = ['fe_grab_pros','cd_mode_come']
    dfCotizaciones = dfCotizaciones[columnas]

    # creamos SEMANA que es el lunes anterior.
    dfCotizaciones['SEMANA'] = dfCotizaciones['fe_grab_pros'].dt.date
    dfCotizaciones['SEMANA'] = pd.to_datetime(dfCotizaciones['SEMANA'])

    # Obtenemos en SEMANA el lunes anterior
    dfCotizaciones['SEMANA'] = dfCotizaciones['SEMANA'].apply(get_previous_monday)

    dfCotizaciones = dfCotizaciones.drop(columns=['fe_grab_pros'])

    # obtenemos las Cotizaciones por SEMANA y MODELO (cd_mode_come)
    dfCotizaciones = dfCotizaciones.groupby(['SEMANA', 'cd_mode_come']).agg(COTIZACIONES=('cd_mode_come', 'count')).reset_index()

    # ordenamos por modelo y semana
    dfCotizaciones = dfCotizaciones.sort_values(by=['cd_mode_come','SEMANA'])

    return dfCotizaciones


def process_data_tasas(dfTasas):
    dfTasas.columns = ['periodo','banrep_colocacion','colocacion','consumo','ordinarios','comercial','tesoreria']

    # MES tiene el primer día del mes.
    dfTasas['SEMANA'] = dfTasas['periodo'].apply(get_previous_monday)

    # Nos interesa solo las tasas de consumo
    dfTasas = dfTasas[['SEMANA','consumo']].copy()

    # RENOMBRAMOS para que nos quede todo bien
    dfTasas.rename(columns={'consumo': 'TASA_CONSUMO'}, inplace=True)

    # asegurarnos que está ordenado
    dfTasas = dfTasas.sort_values(by='SEMANA')

    # # creamos las 12 columnas anteriores
    # for i in range(1, 13):
    #     dfTasas[f'TASA_CONSUMO_{i}'] = dfTasas['TASA_CONSUMO'].shift(i)

    # eliminamos filas en donde haya nulos
    dfTasas = dfTasas.dropna()

    return dfTasas


def process_data_disponibles(dfDisponibles):
    # dejamos solo filas desde el 2015
    dfDisponibles = dfDisponibles.loc[dfDisponibles['SEMANA'] >= '2015-01-01'].copy()

    return dfDisponibles



def join (dfVentas, dfCotizaciones, dfTasas, dfDisponibles):
    dfData = dfVentas.merge(dfCotizaciones, on=['SEMANA','cd_mode_come'], how='left')
    dfData.fillna(0, inplace=True)

    dfData = dfData.merge(dfTasas, on='SEMANA', how='left')
    dfData.fillna(0, inplace=True)


    dfData = dfData.merge(dfDisponibles, on=['SEMANA','cd_mode_come'], how='left')
    dfData.fillna(0, inplace=True)

    # hacemos join de dfVentas consigo misma
    dfData = dfData.merge(dfData, on='cd_mode_come', how='inner')

    # calculamos la distancia 
    dfData['distancia'] = (dfData['SEMANA_x'] - dfData['SEMANA_y']).dt.days/7

    dfData['distancia'] = dfData['distancia'].astype(int)

    # dejamos solo cuando distancia sea menor a 52
    #dfData = dfData.loc[dfData['distancia']>=0]
    #dfData = dfData.loc[dfData['distancia']<=52]

    # dejamos solo las columnas que nos interesan
    dfData = dfData[['SEMANA_x','cd_mode_come','cd_cia_x',	'cd_uneg_cont_x',	'cd_marca_x',	'cd_line_vehi_x', 'distancia','VENTAS_y','COTIZACIONES_y','TASA_CONSUMO_y','disponibles_y']].copy()
    dfData.columns=['SEMANA','cd_mode_come','cd_cia','cd_uneg_cont','cd_marca','cd_line_vehi','distancia','VENTAS','COTIZACIONES','TASA_CONSUMO','DISPONIBLES']

    # dejamos en dfData2 lo único que requerimos para el pivot
    dfData2 = dfData[['SEMANA', 'cd_mode_come', 'distancia', 'VENTAS', 'COTIZACIONES','TASA_CONSUMO','DISPONIBLES']].copy()

    # pivoteamos para obtener las VENTAS anteriores
    dfData2 = dfData2.pivot_table(index=['SEMANA', 'cd_mode_come'], columns='distancia', values=['VENTAS', 'COTIZACIONES','TASA_CONSUMO','DISPONIBLES'], fill_value=0)

    # Aplanar el DataFrame
    dfData2 = dfData2.reset_index()
    dfData2 = dfData2.sort_values(by=['SEMANA', 'cd_mode_come'])
    dfData2 = dfData2.reset_index(drop=True)
    dfData2.columns = ['_'.join(map(str, col)).strip() for col in dfData2.columns.values]

    # renombarmos SEMANA y MODELO
    dfData2 = dfData2.rename(columns={'SEMANA_':'SEMANA', 'cd_mode_come_':'cd_mode_come'})

    # eliminamos las columnas que no podemos usar.  VENTAS_0 es la variable a predecir

    dfData2 = dfData2.drop(['COTIZACIONES_0',
                      'TASA_CONSUMO_0','TASA_CONSUMO_1','TASA_CONSUMO_2', 
                      'DISPONIBLES_0'], axis=1)

    # Obtener el mes del año
    dfData2['Mes'] = dfData2['SEMANA'].dt.month

    # Obtener el número de la semana del año
    dfData2['Semana_del_Año'] = dfData2['SEMANA'].dt.isocalendar().week

    # Obtener el trimestre
    dfData2['Trimestre'] = dfData2['SEMANA'].dt.quarter

    # OJO dejamos solo filas desde el 2015
    dfData2 = dfData2.loc[dfData2['SEMANA']>='2015-01-01'].copy()

    dfData = dfData[['SEMANA',	'cd_mode_come',	'cd_cia',	'cd_uneg_cont',	'cd_marca',	'cd_line_vehi']].copy()

    dfData = dfData2.merge(dfVentas , on=['SEMANA','cd_mode_come'], how='inner')

    dfData = dfData.drop('VENTAS', axis=1)

    return dfData

    #save_to_feather(dfData, "./Data Original/dfData.feather")
    
