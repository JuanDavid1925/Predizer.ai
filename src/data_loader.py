import pandas as pd

def load_data(feather_path):
  """Carga los datos desde un archivo Feather."""
  df = pd.read_feather(feather_path)
  df.reset_index(drop=True, inplace=True)
  return df

def load_data_tasas(feather_path):
   df = pd.read_excel(feather_path)
   return df

def save_to_feather(df, feather_path):
    """Guarda un DataFrame en formato Feather."""
    df.to_feather(feather_path)
    


if __name__ == "__main__":
    dfVentas = load_data("./Data Original/ventas.feather")
    dfLdP = load_data("./Data Original/dfLdP.feather")
    dfCotizaciones = load_data("./Data Original/cotizaciones.feather")
    dfTasas = load_data_tasas('./Data Original/BANREP Historico tasas de interes creditos.xlsx')
    dfDisponibles = load_data("./Data Original/dfDisponibles.feather")
    dfData = load_data("./Data Original/dfData.feather")