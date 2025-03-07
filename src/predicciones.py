import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.regression import *
from src.data_loader import load_data, load_data_tasas
from src.data_processing import fechaMasCercana
from src.data_processing import process_data_precios, join
from src.data_processing import process_data_ventas
from src.data_processing import process_data_precios
from src.data_processing import process_data_cotizaciones
from src.data_processing import process_data_tasas
from src.data_processing import process_data_disponibles
import dagshub
from src.generarArchivoPrediccion import generar_archivo_prediccion
import ast  # Importar módulo para convertir texto a lista


def prediccion(idModelo, fecha_maxima):
    """
    Genera predicciones semanales usando la lista de precios, reentrenando el modelo en cada iteración 
    y utilizando el modelo actualizado para predicciones hasta la fecha máxima.
    """
    fecha_maxima_prediccion = pd.to_datetime(fecha_maxima)

    # Cargar el DataFrame de entrenamiento
    ruta_df = f"Data Training/dfData_{idModelo}.feather"
    dfData = load_data(ruta_df)
    
    # Cargar el DataFrame de parámetros de entrenamiento
    ruta_parametros = "Data Training/ParametrosEntrenamiento.xlsx"
    dfParametros = pd.read_excel(ruta_parametros)

    # Obtener fechas de inicioTraining y fechaDeCorte según el ID
    parametros = dfParametros[dfParametros['nombreArchivo'] == idModelo]
    if parametros.empty:
      raise ValueError(f"No se encontraron parámetros para el ID {idModelo} en ParametrosEntrenamiento.")
    
    fechaFinalTraining = pd.to_datetime(parametros['fechaFinalTraining'].values[0])
    columnas_adicionales_entrenamiento = ast.literal_eval(parametros['columnasAdicionales'].values[0])

    # Obtener la última fecha de entrenamiento
    fecha_actual = fechaFinalTraining + pd.DateOffset(weeks=1)

    # Cargar los datos originales
    dfVentas = load_data("Data Original/ventas.feather")
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfCotizaciones = load_data("Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")

    dfLdP = process_data_precios(dfLdP)

    # Calcular ventas por semana por modelo
    dfVentas = process_data_ventas(dfVentas, dfLdP, fecha_actual, fecha_maxima_prediccion,columnas_adicionales_entrenamiento)

    # Calcular cotizaciones por semana por modelo
    dfCotizaciones = process_data_cotizaciones(dfCotizaciones, fecha_actual, fecha_maxima_prediccion)

    # Calcular tasas por semana
    dfTasas = process_data_tasas(dfTasas, fecha_actual, fecha_maxima_prediccion)

    # Calcular disponibles por semana por modelo
    dfDisponibles = process_data_disponibles(dfDisponibles, fecha_actual, fecha_maxima_prediccion)

    # Hacer el join de los DataFrames
    dfDataPrueba = join(dfVentas, dfCotizaciones, dfTasas, dfDisponibles, fecha_actual, fecha_maxima_prediccion)

    dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
    mlflow.set_experiment("predicciones_PyCaret")

    # Cargar el modelo desde MLflow
    print("VOY A CARGAR EL MODELO:")
    model_path = f"models:/best_model_id_{idModelo}/latest"
    modelo = mlflow.sklearn.load_model(model_path)

    # Inicializar dfTotal antes del ciclo while para acumular todas las semanas
    dfTotal = pd.DataFrame()
    
    while fecha_actual <= fecha_maxima_prediccion:
        modelos_disponibles = fechaMasCercana(fecha_actual, dfLdP)
        dfPrediccion = generar_archivo_prediccion(fecha_actual, modelos_disponibles, idModelo)
        dfPrediccion = dfPrediccion.drop(columns=['SEMANA'])

        columnas_entrenamiento = modelo.feature_names_in_  # Obtener las columnas usadas en el entrenamiento
        
        dfPrediccion = dfPrediccion[columnas_entrenamiento]  # Seleccionar solo esas columnas

        # Hacer predicciones sobre el dfTest filtrado
        setup(data = dfData.drop(columns=['SEMANA']), target='VENTAS_0', session_id=123)
        predictions = predict_model(modelo, data=dfPrediccion)

        predictions['SEMANA'] = fecha_actual

        # Unir predicciones con `dfDataPrueba` para obtener `VENTAS_0` si está disponible
        dfSemana = predictions.merge(dfDataPrueba[['SEMANA', 'cd_mode_come', 'VENTAS_0']], 
                            on=['SEMANA', 'cd_mode_come'], 
                            how='left')
        
        # Seleccionar solo las columnas relevantes
        dfSemana = dfSemana[['SEMANA', 'cd_mode_come', 'VENTAS_0', 'prediction_label']]
        
        # Acumular resultados en dfTotal
        dfTotal = pd.concat([dfTotal, dfSemana], ignore_index=True)

        '''
        # Agregar nuevas filas al conjunto de entrenamiento
        dfTrain = pd.concat([dfTrain, dfTest], ignore_index=True)

        # Formar dfTrain con las fechas obtenidas
        dfTrain = dfTrain[(dfTrain['SEMANA'] >= fechaHistoria) & (dfTrain['SEMANA'] <= fecha_actual)].copy()

        if dfTrain.empty:
            raise ValueError("El conjunto de entrenamiento (dfTrain) está vacío después del filtrado.")
        
        # Reentrenar el modelo con el nuevo conjunto de datos actualizado
        regression_setup = setup(data=dfTrain.drop(columns=['SEMANA']), target='VENTAS_0', session_id=123)
        best_model = compare_models()

        # afinamos el modelo
        tuned_model = tune_model(best_model)

        # Guardar el modelo reentrenado en MLflow
        mlflow.sklearn.log_model(tuned_model, f"best_model_id_{idModelo}")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model_id_{idModelo}"
        mlflow.register_model(model_uri, f"best_model_id_{idModelo}")
        '''
        fecha_actual += pd.DateOffset(weeks=1)

    # Guardar dfTotal en un archivo dentro de 'Data Predictions'
    # Seleccionar solo las columnas relevantes
    dfTotal = dfTotal[['SEMANA', 'cd_mode_come', 'VENTAS_0', 'prediction_label']]
    ruta_predicciones = f"Data Predictions/dfPredicciones_{idModelo}.xlsx"
    dfTotal.to_excel(ruta_predicciones, index=False)
    
    print("✅ Predicciones completadas hasta la fecha máxima.")