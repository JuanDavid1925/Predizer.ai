import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.regression import *
from src.data_loader import load_data
from src.data_processing import fechaMasCercana
from src.data_processing import process_data_precios
import dagshub

def prediccion(idModelo, fecha_maxima):
    """
    Genera predicciones semanales usando la lista de precios, reentrenando el modelo en cada iteración 
    y utilizando el modelo actualizado para predicciones hasta la fecha máxima.
    """
    fecha_maxima_prediccion = pd.to_datetime(fecha_maxima)

    # Cargar el DataFrame de entrenamiento
    ruta_df = f"Data Training/dfData_{idModelo}.feather"
    dfTrain = load_data(ruta_df)
    
    if dfTrain is None or dfTrain.empty:
        raise ValueError(f"El archivo {ruta_df} no existe o está vacío.")
    
    # Cargar el DataFrame de parámetros de entrenamiento
    ruta_parametros = "Data Training/ParametrosEntrenamiento.xlsx"
    dfParametros = pd.read_excel(ruta_parametros)

    # Obtener fechas de inicioTraining y fechaDeCorte según el ID
    parametros = dfParametros[dfParametros['nombreArchivo'] == idModelo]
    if parametros.empty:
      raise ValueError(f"No se encontraron parámetros para el ID {idModelo} en ParametrosEntrenamiento.")
    
    fechaHistoria = pd.to_datetime(parametros['fechaInicioTraining'].values[0])
    fechaBase = pd.to_datetime(parametros['fechaDeCorte'].values[0])

    # Obtener la última fecha de entrenamiento
    fecha_actual = fechaBase + pd.DateOffset(weeks=1)
    dfLdP = load_data("Data Original/dfLdP.feather")
    dfLdP = process_data_precios(dfLdP)

    dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
    
    while fecha_actual <= fecha_maxima_prediccion:
        modelos_disponibles = fechaMasCercana(fecha_actual, dfLdP)
        dfTest = dfTrain[dfTrain['cd_mode_come'].isin(modelos_disponibles)].copy()
        dfTest = dfTest.drop(columns=['SEMANA'])
        
        #dfTest = dfTest.drop(columns=['VENTAS_0'], errors='ignore')

        # Cargar el modelo desde MLflow
        print("VOY A CARGAR EL MODELO:")
        model_path = f"models:/best_model_id_{idModelo}/latest"
        modelo = mlflow.pyfunc.load_model(model_path)
        
        # Hacer predicciones sobre el dfTest filtrado
        predictions = predict_model(modelo, data=dfTest.drop(columns=['VENTAS_0']))

        dfTotal = pd.merge(predictions, dfTrain, left_index=True, right_index=True)
        
        dfTest['SEMANA'] = fecha_actual

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
        
        fecha_actual += pd.DateOffset(weeks=1)

    # Guardar dfTotal en un archivo dentro de 'Data Predictions'
    dfTotal[['SEMANA','cd_mode_come_x','VENTAS_0','prediction_label']]
    ruta_predicciones = f"Data Predictions/dfPredicciones_{idModelo}.xlsx"
    dfTotal.to_excel(ruta_predicciones, index=False)
    
    print("✅ Predicciones completadas hasta la fecha máxima.")