import pandas as pd
import pandasql as psql
from pycaret.regression import *
import mlflow
import mlflow.sklearn  # Para modelos de scikit-learn (PyCaret usa scikit-learn internamente)
import dagshub
#from src.data_processing import fechaMasCercana
#from sklearn.metrics import mean_absolute_percentage_error


def entrenamiento(dfData, total_cotizaciones, total_tasas, total_disponibles, historia, horizonte, semana_inicial):
  # lista de semanas disponibles
  listaSEMANA = dfData['SEMANA'].unique()
  # ============================

  sin_cero = False
  fechaBase = pd.to_datetime(semana_inicial)
  comentario = "MV"  # VV varias variables (y no solo SEMANA, modelo) como son columnas = ['fe_pedi_cli','cd_mode_come','cd_cia','cd_uneg_cont','cd_sucu','cd_marca','cd_line_vehi']
                   # MV menos variables:  es decir, dejar por ejemplo menos variables de COTIZACIÓN (ultimos 3 meses), tasas (3 meses),  DISPONIBLE (3 MESES)

  
  # Eliminar todas las columnas de COTIZACIONES, DISPONIBLES, TASA_CONSUMO que sobran
  columns_to_drop1 = [col for col in dfData.columns if col.startswith('COTIZACIONES') and int(col.split('_')[1]) > total_cotizaciones]
  columns_to_drop2 = [col for col in dfData.columns if col.startswith('DISPONIBLES') and int(col.split('_')[1]) > total_disponibles]
  columns_to_drop3 = [col for col in dfData.columns if col.startswith('TASA') and int(col.split('_')[2]) > total_tasas]
  #columns_to_drop4 = [col for col in dfData.columns if col.startswith('VENTAS') and int(col.split('_')[1]) > total_ventas]
  columns_to_drop = columns_to_drop1 + columns_to_drop2 + columns_to_drop3 

  dfData = dfData.drop(columns=columns_to_drop)


  # Configurar experimento en MLflow
  dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
  mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
  mlflow.set_experiment("Prediccion_Ventas4")

  # Definir la fecha de historia y horizonte
  fechaHistoria = fechaBase - pd.DateOffset(years=historia)
  # Conjunto de entrenamiento con todos los modelos de vehículos
  dfTrain = dfData[(dfData['SEMANA'] >= fechaHistoria) & (dfData['SEMANA'] < fechaBase)].copy()
  dfTest = dfData[(dfData['SEMANA'] >= fechaBase) & (dfData['SEMANA'] < fechaBase + pd.DateOffset(weeks=horizonte))].copy()

  # eliminamos semana
  dfTrain = dfTrain.drop(columns=['SEMANA'])
  dfTest = dfTest.drop(columns=['SEMANA'])

  print(dfTest)
  
  with mlflow.start_run(run_name=f"Hist_{historia}_Hor_{horizonte}"):
    # Inicializar la configuración de regresión
    regression_setup = setup(data=dfTrain, target='VENTAS_0', session_id=123)

    # Comparar modelos y seleccionar el mejor
    best_model = compare_models()
    best_model_name = str(best_model)
    

    # afinamos el modelo
    tuned_model = tune_model(best_model)
    

    # Hacer predicciones sobre el dfTest filtrado
    predictions = predict_model(tuned_model, data=dfTest.drop(columns=['VENTAS_0']))

    
                    
    dfTotal = pd.merge(predictions, dfData, left_index=True, right_index=True)
    dfTotal[['SEMANA','cd_mode_come_x','VENTAS_0','prediction_label']]
    dfTotal['historia'] = historia
    dfTotal['sin_cero'] = False

    # Calcular precisión del modelo
    #precision = 1 - mean_absolute_percentage_error(dfTotal['VENTAS_0'], dfTotal['prediction_label'])
    errorAbsoluto = ((dfTotal['VENTAS_0'] - dfTotal['prediction_label'].round().astype(int)).abs()).mean()
    metrics = pull()
  

    # Guardar el mejor modelo en MLflow
    mlflow.sklearn.log_model(best_model, f"best_model_historia_{historia}")

    # Loggear parámetros en MLflow
    mlflow.log_param("Historia", historia)
    mlflow.log_param("Horizonte", horizonte)
    mlflow.log_param("Semana_inicial", semana_inicial)
    mlflow.log_param("Mejor modelo", best_model_name)

    # Loggear métricas en MLflow
    if metrics is not None and not metrics.empty:
          mlflow.log_metric("MAE", pull().loc["Mean", "MAE"] if "MAE" in metrics.columns else 0)
          mlflow.log_metric("MSE", pull().loc["Mean","MSE"] if "MSE" in metrics.columns else 0)
          mlflow.log_metric("R2", pull().loc["Mean","R2"] if "R2" in metrics.columns else 0)
          mlflow.log_metric("RMSE", pull().loc["Mean","RMSE"] if "RMSE" in metrics.columns else 0)
          mlflow.log_metric("RMSLE", pull().loc["Mean","RMSLE"] if "RMSLE" in metrics.columns else 0)
          mlflow.log_metric("MAPE", pull().loc["Mean","MAPE"] if "MAPE" in metrics.columns else 0)
          mlflow.log_metric("Error absoluto", errorAbsoluto)

    
    # Imprimir ventas reales y predichas organizadas por modelo de vehículo
    for modelo in dfTotal['cd_mode_come_x'].unique():
          dfModelo = dfTotal[dfTotal['cd_mode_come_x'] == modelo]
          print(f"\nResultados para el modelo: {modelo}")
          print(dfModelo[['SEMANA', 'VENTAS_0', 'prediction_label']])
    
    mlflow.end_run()
    




    


