import pandas as pd
import os


def load_data(feather_path):
  """Carga los datos desde un archivo Feather."""
  df = pd.read_feather(feather_path)
  df.reset_index(drop=True, inplace=True)
  return df

'''
def load_data(file_path):
  # Obtener la extensión del archivo
  ext = os.path.splitext(file_path)[1].lower()

  # Leer el archivo según su extensión
  if ext == ".xlsx":
      df = pd.read_excel(file_path)
  elif ext == ".csv":
      df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1', delimiter='|')
  else:
      raise ValueError("Formato no soportado. Usa un archivo .xlsx o .csv")
  
  # Obtener el nombre base sin extensión
  base_name = os.path.splitext(os.path.basename(file_path))[0]

  # Definir la ruta de guardado en formato .feather
  feather_path = os.path.join(os.path.dirname(file_path), f"{base_name}.feather")

  # Guardar en formato Feather
  df.to_feather(feather_path)

  # Leer el archivo .feather para comprobar su correcto guardado
  df = pd.read_feather(feather_path)
  df.reset_index(drop=True, inplace=True)
  return df
'''

def load_data_tasas(feather_path):
   df = pd.read_excel(feather_path)
   return df

def save_to_feather(df, feather_path):
    """Guarda un DataFrame en formato Feather."""
    df.to_feather(feather_path)
    


if __name__ == "__main__":
    dfVentas = load_data("Data Original/ventas 2024 12 10.xlsx")
    dfLdP = load_data("Data Original/listaPrecios.xlsx")
    dfCotizaciones = load_data("Data Original/cotizaciones.xlsx")
    dfTasas = load_data_tasas("Data Original/BANREP Historico tasas de interes creditos.xlsx")
    dfDisponibles = load_data("Data Original/dfDisponibles.feather")