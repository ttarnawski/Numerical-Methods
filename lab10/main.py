import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(x, np.ndarray):
        return None
    return P.polyfromroots(x)

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(a, np.ndarray):
        return None
    else:
        for i in range (len(a)):
            a = a + np.random.random_sample() * 10**(-10)
            roots = P.polyroots(a)
        return a, roots

# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(wsp, np.ndarray):
        return None
    else:
        frobenus_matrix = np.zeros((len(wsp), len(wsp)))
        for row in range(len(wsp)):
            for col in range(len(wsp)):
                if row != len(wsp) - 1:
                    if row + 1 == col:
                        frobenus_matrix[row][col] = 1
                else:
                    frobenus_matrix[row][col] = -wsp[col]
        return frobenus_matrix, np.linalg.eigvals(frobenus_matrix), scipy.linalg.schur(frobenus_matrix), P.polyfromroots(wsp)




