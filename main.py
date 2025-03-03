from src.generarArchivoInicialTraining import generar_archivo_inicial_training
from src.entrenarModelosPyCaret import entrenar_modelo
from src.predicciones import prediccion

def main():
    
    generar_archivo_inicial_training(horizonte=4, 
                                     historia_ventas=12, 
                                     historia_cotizaciones=12, 
                                     historia_tasas=12, 
                                     historia_disponibles=12,
                                     fechaInicioTraining='2015-01-01',
                                     fechaDeCorte = '2024-07-01',  # YYYY-MM-DD, es la fecha que señala el final de trainig y el inicio de test
                                     columnasAdicionales= ['cd_cia', 'cd_uneg_cont', 'cd_marca', 'cd_line_vehi']) 
    
    entrenar_modelo(idData = 1)

    prediccion(idModelo = 1 , fecha_maxima = '2024-07-15')

    
if __name__ == "__main__":
    main()