import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None
    A = np.random.randint(0, 9, size=(m, m))
    b = np.random.randint(0, 9, size=(m))
    for row in range(m):
        sum = 0
        for column in range(m):
            if row != column:
                sum += abs(A[row][column])
            else:
                if A[row][column] <= sum:
                    A[row][column] += sum
    return A, b


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray) or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        return None
    diag = np.diag(np.diag(A))
    sum_diag = np.sum(diag)
    A = A - diag
    for m in range(len(A)):
        if np.sum(A[m]) >= sum_diag:
            return False
    return True


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None
    A = np.random.randint(0, 100, size=(m, m))
    b = np.random.randint(0, 9, size=(m))
    A_transpose = np.transpose(A)
    for row in range(m):
        for col in range(m):
            if row > col:
                A[row][col] = A_transpose[row][col]
    return A, b


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray) or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        return None
    for row in range(len(A)):
        for col in range(len(A)):
            if A[row][col] != A[col][row]:
                return False
    return True


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(x_init, np.ndarray) or not isinstance(epsilon, float) or not isinstance(maxiter, int):
        return None
    if A.shape[0] != A.shape[1] or x_init.shape != b.shape:
        return None
    D = np.diag(np.diag(A))
    LU = A - D
    D_inv = np.diag(1 / np.diag(D))
    resid = []
    for i in range(maxiter):
        x_new = np.dot(D_inv, b - np.dot(LU, x_init))
        r_norm = np.linalg.norm(x_new - x_init)
        resid.append(r_norm)
        if r_norm < epsilon:
            return x_new, resid
        x_init = x_new
    return x_init, resid


def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None
    else:
        A = np.random.randint(0, 100, size=(m, m))
        b = np.random.randint(0, 100, size=(m,))
    return A, b


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b
      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
      return None
    else:
      r = b - A @ x
      return np.linalg.norm(r)