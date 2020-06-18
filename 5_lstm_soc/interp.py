import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


def pchip_random(data: np.ndarray, datapoints = 23) -> np.ndarray:
    """Interpolação aleatória gerada com o polinômio interpolador do pchip
    
    Args:
        data (np.ndarray): Dados a serem inserpolados
        datapoints (int): Quantidade de valores geradas em um intervalo
    Returns:
        np.ndarray: Valores interpolados
    """
    interpolated_values = []
    for i in range(0, len(data) - 1):
        today = data[i]
        tomorrow = data[i + 1]
        
        if today == tomorrow:
            continue

        fnc = PchipInterpolator([0, 1], [today, tomorrow])
        elements = fnc(np.arange(0, 1, 1 / (datapoints + 1)))
        interpolated_values.extend(elements)
    return np.array(interpolated_values)


def interp1d_random(data: np.ndarray, datapoints = 23):
    """Interpolação aleatória gerada com elementos randômicos inteiros em um intervalo
    
    Args:
        data (np.ndarray): Dados a serem inserpolados
        datapoints (int): Quantidade de valores geradas em um intervalo
    Returns:
        np.ndarray: Valores interpolados
    """
    
    interpolated_values = []
    
    for i in range(0, len(data) - 1):
        today = data[i]
        tomorrow = data[i + 1]
        
        if today == tomorrow:
            continue # Neste caso, não há o que ser feito
        
        if today > tomorrow:
            elements = np.random.randint(tomorrow, today, datapoints).tolist()
        else:
            elements = np.random.randint(today, tomorrow, datapoints).tolist()

        elements.insert(0, today)
        elements.append(tomorrow)
        interpolated_values.extend(elements)
    return np.array(interpolated_values)
