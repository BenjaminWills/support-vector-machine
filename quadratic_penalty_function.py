import scipy.optimize as optimise
import numpy as np
from typing import Callable,List

def positive_part(x:float) -> float:
    """Returns x if x is positive, and 0 otherwise

    Parameters
    ----------
    x : float
    Returns
    -------
    float
    """
    if x > 0: return x 
    else: return 0

def get_positive_array(points:np.array) -> np.array:
    """Applies positive_part to a whole array elementwise

    Parameters
    ----------
    points : np.array
    Returns
    -------
    np.array
    """
    return np.array(
        [positive_part(element)
        for element in points] 
    )

def get_function_norm(x:float,functions:List[Callable]) -> float:
    """Gets the norm of an array filled with function values evaluated at
    the point x.

    Parameters
    ----------
    x : float
    functions : List[Callable]
        A list of functions

    Returns
    -------
    float
    """
    return np.linalg.norm(
        np.array([
            function(x) 
            for function in functions
        ])
    )

def get_positive_function_norm(x:float,functions:List[Callable])->float:
    """Gets the norm of an array filled with function values evaluated at
    the point x, including only the positive values.

    Parameters
    ----------
    x : float
    functions : List[Callable]
        A list of functions

    Returns
    -------
    float
    """
    return np.linalg.norm(
        np.array([
            positive_part(function(x)) 
            for function in functions
        ])
    )

def quadratic_penalty_function(
    x:np.array,
    gamma:float,
    objective_function:Callable,
    equality_constraints:List[Callable],
    inequality_constraints:List[Callable]) -> float:
    """Quadratic penalty function for the algorithm.

    Parameters
    ----------
    x : np.array
        _description_
    gamma : float
        _description_
    objective_function : Callable
    equality_constraints : List[Callable]
        list of equality constraints
    inequality_constraints : List[Callable]
        list of inequality constraints

    Returns
    -------
    float
    """
    penalty_value = gamma/2
    return objective_function(x) + penalty_value * (get_positive_function_norm(x,inequality_constraints)) ** 2 + penalty_value * (get_function_norm(x,equality_constraints)) ** 2

def truncation(x:np.array,max_norm:float) -> np.array:
    """To ensure convergence, maximising the norm of any vector.

    Parameters
    ----------
    x : np.array
    max_norm : float

    Returns
    -------
    np.array
    """
    norm = np.linalg.norm(x) 
    if norm <= max_norm:
        return x
    else:
        return max_norm * x / norm


def quadratic_penalty_method(
    initial_point:np.array,
    initial_gamma:float,
    epsilon:float,
    c:float,
    delta:float,
    objective_function:Callable,
    equality_constraints:List[Callable],
    inequality_constraints:List[Callable]) -> np.array:
    """_summary_

    Parameters
    ----------
    initial_point : np.array
    initial_gamma : float
        _description_
    epsilon : float
        Stopping criterion for equality and inequality constraints
    c : float
        Stopping criterion for gamma
    delta : float
        gamma multiplier at each iteration
    objective_function : Callable
    equality_constraints : List[Callable]
    inequality_constraints : List[Callable]

    Returns
    -------
    np.array
        constrained minimum
    """
    x = initial_point
    gamma = initial_gamma
    iterations = 1
    while get_positive_function_norm(x,inequality_constraints) > epsilon or\
         get_function_norm(x,equality_constraints) > epsilon or\
             gamma < c:
        
        quadratic_penalty = lambda x: quadratic_penalty_function(
            x,
            gamma,
            objective_function,
            equality_constraints,
            inequality_constraints)
        
        x = optimise.minimize(quadratic_penalty,x)['x']
        iterations += 1
        gamma *= delta
    print(iterations)
    return x