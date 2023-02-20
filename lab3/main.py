import numpy as np

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, List, np.ndarray)) == False or isinstance(v_aprox, (int, float, List, np.ndarray)) == False:
        return np.NaN
    if isinstance(v, (List, np.ndarray)) and isinstance(v_aprox, (List, np.ndarray)) and len(v) != len(v_aprox):
        return np.NaN
    
    return abs(np.array(v) - np.array(v_aprox))


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, List, np.ndarray)) == False or isinstance(v_aprox, (int, float, List, np.ndarray)) == False:
        return np.NaN
    if isinstance(v, (List, np.ndarray)) and isinstance(v_aprox, (List, np.ndarray)) and (len(v) != len(v_aprox) or 0 in v):
        return np.NaN
    if isinstance(v, (int, float)) and v == 0:
        return np.NaN

    return abs((np.array(v) - np.array(v_aprox)) / v)


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) == False or isinstance(c, (int, float)) == False: 
        return np.NaN
    
    b = 2**n
    P1 = b - b + c
    P2 = b + c - b
    return abs(P1 - P2)


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) == False or isinstance(x, (int, float)) == False or n <= 0:
        return np.NaN
    
    exp_aprox = 0
    for i in range(n):
        exp_aprox += 1 / scipy.math.factorial(i) * (x**i)
    return exp_aprox


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) == False or isinstance(x, (int, float)) == False or k < 0:
        return np.NaN

    if k == 1 or k == 0:
        return np.cos(k * x)
    coskx = 2 * np.cos(x) * coskx1(k - 1, x) - coskx1(k - 2, x)
    return coskx
    

def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) == False or isinstance(x, (int, float)) == False or k < 0:
        return np.NaN

    def cos2(j: int, x: Union[int, float]) -> float:
        if j == 1 or j == 0:
            return np.cos(j * x)
        return np.cos(x) * cos2(j - 1, x) - np.sin(x) * sin2(j - 1, x)

    def sin2(j: int, x: Union[int, float]) -> float:
        if j == 1 or j == 0:
            return np.sin(j * x)
        return np.sin(x) * cos2(j - 1, x) + np.cos(x) * sin2(j - 1, x)

    return cos2(k, x), sin2(k, x)


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) == False or n == 0:
        return np.NaN
    
    pi_aprox = 0
    for i in range(1, n + 1):
        pi_aprox += 1 / i**2
    return np.sqrt(6 * pi_aprox)