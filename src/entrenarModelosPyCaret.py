import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.regression import *
from src.data_loader import load_data
import dagshub

def entrenar_modelo(idData):
  """
  Realiza el entrenamiento del modelo utilizando el DataFrame de entrenamiento correspondiente al ID proporcionado
  y guarda los parámetros en MLflow.
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
    
  # eliminamos semana
  dfData = dfData.drop(columns=['SEMANA'])

  # Configurar experimento en MLflow
  dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
  mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
  mlflow.set_experiment("Entrenamiento_PyCaret")

  with mlflow.start_run(run_name=f"Entrenamiento_ID_{idData}"):
        # Registrar los parámetros en MLflow
        for col in parametros.columns:
            mlflow.log_param(col, parametros[col].values[0])

        # Configurar PyCaret
        regression_setup = setup(data=dfData, target='VENTAS_0', session_id=123)

        # Comparar y seleccionar el mejor modelo
        best_model = compare_models()
        best_model_name = str(best_model)

        # afinamos el modelo
        tuned_model = tune_model(best_model)

        metrics = pull()
        # Loggear métricas en MLflow
        if metrics is not None and not metrics.empty:
          mlflow.log_metric("MAE", pull().loc["Mean", "MAE"] if "MAE" in metrics.columns else 0)
          mlflow.log_metric("MSE", pull().loc["Mean","MSE"] if "MSE" in metrics.columns else 0)
          mlflow.log_metric("R2", pull().loc["Mean","R2"] if "R2" in metrics.columns else 0)
          mlflow.log_metric("RMSE", pull().loc["Mean","RMSE"] if "RMSE" in metrics.columns else 0)
          mlflow.log_metric("RMSLE", pull().loc["Mean","RMSLE"] if "RMSLE" in metrics.columns else 0)
          mlflow.log_metric("MAPE", pull().loc["Mean","MAPE"] if "MAPE" in metrics.columns else 0)
        
        # Registrar el modelo en MLflow
        model_path = f"best_model_id_{idData}"
        mlflow.sklearn.log_model(tuned_model, model_path)
        # Registrar el modelo en MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model_id_{idData}"
        mlflow.register_model(model_uri, f"best_model_id_{idData}")
        mlflow.log_param("Mejor modelo", best_model_name)

        print(f"✅ Entrenamiento completado para ID {idData}. Mejor modelo registrado en MLflow.")