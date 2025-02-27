from src.data_loader import load_data
from src.data_loader import load_data_tasas
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
from src.data_processing import join
from src.analysis import entrenamiento
from src.cargarModelo import cargarModelo
from src.generarArchivoInicialTraining import generar_archivo_inicial_training

def main():
    '''
    dfVentas = load_data("Data Original/ventas.feather")
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfCotizaciones = load_data("Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")


    dfLdP = process_data_precios(dfLdP)
    dfVentas = process_data_ventas(dfVentas, dfLdP)
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones)
    dfTasas = process_data_tasas(dfTasas)
    dfDisponibles = process_data_disponibles(dfDisponibles)

    join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles)

    dfData = load_data("Data Original/dfData.feather")
    
    #entrenamiento(dfData, 12, 12, 12, 5, 4, '2024-11-01')

    #print(cargarModelo(dfData, '2024-11-01', 4))
    '''
    generar_archivo_inicial_training(4, 12, 12, 12, 12,'2024-07-01')
    
if __name__ == "__main__":
    main()