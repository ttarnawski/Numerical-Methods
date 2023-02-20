import math
import numpy as np

def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if isinstance(r, float) is True and isinstance(h, float) is True:
        if r > 0 and h > 0:
            return 2 * math.pi * r * r + 2 * math.pi * r * h
        else:
            return np.NaN
    else:
        return np.NaN

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if isinstance(n, int) is True:
        if n <= 0:
            return None
        else:        
            if n == 1:
                return [1]
            fib_vector = np.array([1, 1])
            for n in range(2, n):
                fib_vector = np.append(fib_vector, fib_vector[n - 1] + fib_vector[n - 2])
            return fib_vector
    else:
        return None

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a,1,-a],[0,1,1],[-a,a,1]])
    if a == 0:
        return np.NaN, np.transpose(M), np.linalg.det(M)
    else:
        return np.linalg.inv(M), np.transpose(M), np.linalg.det(M)

        

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if isinstance(m, int) is True and isinstance(n, int) is True:
        if m > 0 and n > 0:
            M = np.zeros((m, n))
            for i in range(0, m):
                for j in range(0, n):
                    if i > j:
                        M[i][j] = i
                    else:
                        M[i][j] = j
            return M
        else:
            return None
    else:
        return None