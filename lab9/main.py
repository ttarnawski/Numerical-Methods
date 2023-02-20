import numpy as np
import scipy
import pickle
import typing
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, float) or not isinstance(iteration, int):
        return None
    elif a > b:
        return None 
    elif f(a) * f(b) < 0:
        for it in range(iteration):
            x1 = (a + b) / 2
            if abs(f(x1)) < epsilon:
                return x1, it
            elif f(a) * f(x1) < 0:
                b = x1
            else:
                a = x1
        return x1, it
    else:
        return None       


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, float) or not isinstance(iteration, int):
        return None
    elif a > b:
        return None
    elif f(a) * f(b) < 0:
        for it in range(iteration):
            xn = a - (f(a) * (b - a)) / (f(b) - f(a))
            if abs(f(xn)) < epsilon:
                return xn, it
            elif f(a) * f(xn) < 0:
                b = xn
            else:
                a = xn
        return xn, it
    else:
        return None


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, float) or not isinstance(iteration, int):
        return None
    elif a > b:
        return None
    elif f(a) * f(b) < 0:
        if f(a) * ddf(a) > 0:
            xn_1 = a
        else:
            xn_1 = b
        for it in range(iteration):
            xn = xn_1 - f(xn_1) / df(xn_1)
            if abs(xn - xn_1) < epsilon:
                return xn, it
            xn_1 = xn
        return xn, it
    else:
        return None
    
            

