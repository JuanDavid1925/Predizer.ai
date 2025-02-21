import pandas as pd
import pandasql as psql
from pycaret.regression import *
import mlflow
import mlflow.sklearn  # Para modelos de scikit-learn (PyCaret usa scikit-learn internamente)
import dagshub


def entrenamiento(dfData):
  # lista de semanas disponibles
  listaSEMANA = dfData['SEMANA'].unique()
  # ============================

  sin_cero = False
  fechaBase = pd.to_datetime('2024-11-01')
  comentario = "MV"  # VV varias variables (y no solo SEMANA, modelo) como son columnas = ['fe_pedi_cli','cd_mode_come','cd_cia','cd_uneg_cont','cd_sucu','cd_marca','cd_line_vehi']
                   # MV menos variables:  es decir, dejar por ejemplo menos variables de COTIZACIÓN (ultimos 3 meses), tasas (3 meses),  DISPONIBLE (3 MESES)

  # Eliminar todas las columnas con de COTIZACIONES, DISPONIBLES, TASA_CONSUMO que sobran
  columns_to_drop1 = [col for col in dfData.columns if col.startswith('COTIZACIONES') and int(col.split('_')[1]) > 12]
  columns_to_drop2 = [col for col in dfData.columns if col.startswith('DISPONIBLES') and int(col.split('_')[1]) > 12]
  columns_to_drop3 = [col for col in dfData.columns if col.startswith('TASA') and int(col.split('_')[2]) > 12]
  columns_to_drop = columns_to_drop1 + columns_to_drop2 + columns_to_drop3

  dfData = dfData.drop(columns=columns_to_drop)

  # Definir el experimento en MLflow
  dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
  mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
  mlflow.set_experiment("Prediccion_Ventas2")

  for historia in [1,2,3,4,5]:
    with mlflow.start_run(run_name=f"Hist_{historia}"):

        # calculamos la fecha de historia, como la fecha base MENOS lo que diga historia
        fechaHistoria = fechaBase - pd.DateOffset(years=historia)

        # creamos TRAINING como antes de noviembre del 2024
        dfTrain = dfData[(dfData['SEMANA'] >= fechaHistoria) & (dfData['SEMANA'] < fechaBase)]

        # y TEST es desde el 1 de febrero
        dfTest = dfData[dfData['SEMANA'] >= fechaBase]

        # eliminamos semana
        dfTrain = dfTrain.drop(columns=['SEMANA'])

        # Inicializar la configuración de regresión
        regression_setup = setup(data=dfTrain, target='VENTAS_0', session_id=123)

        # Comparar modelos y seleccionar el mejor
        best_model = compare_models()
        dfComparaModelos = pull()
        #dfComparaModelos.to_excel('dfComparaModelos_' + str(historia) + '_' + str(sin_cero) + '.xlsx')

        # Guardar el mejor modelo en MLflow
        mlflow.sklearn.log_model(best_model, f"best_model_historia_{historia}")

        # afinamos el modelo
        tuned_model = tune_model(best_model)
        dfTuneModel = pull()
        #dfTuneModel.to_excel('dfTuneModel' + str(historia) + '_' + str(sin_cero) + '.xlsx')

        # eliminamos semana
        dfTest = dfTest.drop(columns=['SEMANA'])

        # Hacer predicciones sobre el dfTest filtrado
        predictions = predict_model(tuned_model, data=dfTest.drop(columns=['VENTAS_0']))

        dfTotal = pd.merge(predictions, dfData, left_index=True, right_index=True)
        dfTotal[['SEMANA','cd_mode_come_x','VENTAS_0','prediction_label']]
        dfTotal['historia'] = historia
        dfTotal['sin_cero'] = True
        #dfTotal.to_excel('dfTotal_' + comentario + '_' + str(historia) + '_' + str(sin_cero) +'.xlsx' )


        # Loggear parámetros en MLflow
        mlflow.log_param("historia", historia)
        mlflow.log_param("sin_cero", sin_cero)

        # Loggear métricas en MLflow 
        metrics = pull()  # Obtiene métricas del mejor modelo
        mlflow.log_metric("MAE", metrics.loc["Mean", "MAE"])
        mlflow.log_metric("MSE", metrics.loc["Mean","MSE"])
        mlflow.log_metric("R2", metrics.loc["Mean","R2"])
        mlflow.log_metric("RMSE", metrics.loc["Mean","RMSE"])
        mlflow.log_metric("RMSLE", metrics.loc["Mean","RMSLE"])
        mlflow.log_metric("MAPE", metrics.loc["Mean","MAPE"])


        # Finalizar el experimento
        mlflow.end_run()
