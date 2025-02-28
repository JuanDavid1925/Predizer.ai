import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.regression import *
from src.data_loader import load_data
import dagshub

def entrenar_modelo(idData):
  """
  Realiza el entrenamiento del modelo utilizando el DataFrame de entrenamiento correspondiente al ID proporcionado.
  """

  # Cargar el DataFrame de entrenamiento
  ruta_df = f"Data Training/dfData_{idData}.feather"
  dfData = load_data(ruta_df)

  if dfData is None or dfData.empty:
        raise ValueError(f"El archivo {ruta_df} no existe o está vacío.")


  # Cargar el DataFrame de parámetros de entrenamiento
  ruta_parametros = "Data Training/ParametrosEntrenamiento.xlsx"
  dfParametros = pd.read_excel(ruta_parametros)

  # Obtener fechas de inicioTraining y fechaDeCorte según el ID
  parametros = dfParametros[dfParametros['nombreArchivo'] == idData]
  if parametros.empty:
      raise ValueError(f"No se encontraron parámetros para el ID {idData} en ParametrosEntrenamiento.")
    
  fechaHistoria = pd.to_datetime(parametros['fechaInicioTraining'].values[0])
  fechaBase = pd.to_datetime(parametros['fechaDeCorte'].values[0])

  # Formar dfTrain con las fechas obtenidas
  dfTrain = dfData[(dfData['SEMANA'] >= fechaHistoria) & (dfData['SEMANA'] < fechaBase)].copy()

  if dfTrain.empty:
        raise ValueError("El conjunto de entrenamiento (dfTrain) está vacío después del filtrado.")

  # eliminamos semana
  dfTrain = dfTrain.drop(columns=['SEMANA'])

  # Configurar experimento en MLflow
  dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
  mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
  mlflow.set_experiment("Entrenamiento_PyCaret")

  with mlflow.start_run(run_name=f"Entrenamiento_ID_{idData}"):
        # Configurar PyCaret
        regression_setup = setup(data=dfTrain, target='VENTAS_0', session_id=123)
        
        # Comparar y seleccionar el mejor modelo
        best_model = compare_models()
        best_model_name = str(best_model)

        # afinamos el modelo
        tuned_model = tune_model(best_model)
        
        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(best_model, f"best_model_id_{idData}")
        mlflow.log_param("Mejor modelo", best_model_name)
        
        print(f"✅ Entrenamiento completado para ID {idData}. Mejor modelo registrado en MLflow.")