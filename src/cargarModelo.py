import mlflow
#from src.data_loader import load_data
import pandas as pd
import dagshub

def cargarModelo(dfData,semana_inicial, horizonte):
  dagshub.init(repo_owner='JuanDavid1925', repo_name='Predizer.ai', mlflow=True)
  mlflow.set_tracking_uri("https://dagshub.com/JuanDavid1925/Predizer.ai.mlflow")
  logged_model = "models:/Prediccion/latest"
  # Cargar el conjunto de datos
  #dfData = load_data("Data Original/dfData.feather")

  # Construir conjunto de prueba con semanas futuras
  dfTest = construir_dfTest(dfData, semana_inicial, horizonte)

  # Load model as a PyFuncModel.
  loaded_model = mlflow.pyfunc.load_model(logged_model)

  # Predict on a Pandas DataFrame.
  predicciones = loaded_model.predict(pd.DataFrame(dfTest))

  return predicciones


def construir_dfTest(dfData, semana_inicial, horizonte):
    """
    Construye un DataFrame de prueba para predicciones futuras.
    
    Parámetros:
    - dfData: DataFrame de datos históricos.
    - semana_inicial: Fecha desde la cual se quiere predecir.
    - horizonte: Cantidad de semanas a predecir.

    Retorna:
    - dfTest con las semanas futuras y las variables necesarias para el modelo.
    """
    
    # Convertir semana inicial a datetime
    fechaBase = pd.to_datetime(semana_inicial)

    # Obtener modelos de vehículos presentes en los datos históricos
    modelos_disponibles = dfData['cd_mode_come'].unique()

    # Crear las semanas futuras en el horizonte de predicción
    semanas_futuras = [fechaBase + pd.DateOffset(weeks=i) for i in range(horizonte)]

    # Crear un DataFrame combinando semanas y modelos de vehículos
    dfTest = pd.DataFrame([(semana, modelo) for semana in semanas_futuras for modelo in modelos_disponibles], 
                          columns=['SEMANA', 'cd_mode_come'])

    # Seleccionar solo las características relevantes (excluir ventas futuras)
    columnas_predictoras = [col for col in dfData.columns if col not in ['VENTAS_0', 'SEMANA']]

    # Filtrar solo columnas numéricas
    columnas_numericas = dfData.select_dtypes(include=['number']).columns.tolist()

    # Asegurar que solo se usen columnas numéricas en el cálculo del promedio
    dfHistorico = dfData.groupby('cd_mode_come')[columnas_numericas].mean().reset_index()

    # Unir las características históricas con las nuevas semanas futuras
    dfTest = dfTest.merge(dfHistorico, on='cd_mode_come', how='left')

    return dfTest

