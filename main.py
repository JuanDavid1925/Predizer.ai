from src.data_loader import load_data
from src.data_loader import load_data_tasas
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
from src.data_processing import join
from src.analysis import entrenamiento


def main():
    dfVentas = load_data("Data Original/ventas.feather")
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfCotizaciones = load_data("Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")


    dfVentas = process_data_ventas(dfVentas)
    dfLdP = process_data_precios(dfLdP)
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones)
    dfTasas = process_data_tasas(dfTasas)
    dfDisponibles = process_data_disponibles(dfDisponibles)

    join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles)


    dfData = load_data("Data Original/dfData.feather")

    entrenamiento(dfData)


if __name__ == "__main__":
    main()