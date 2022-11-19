from quadratic_penalty_function import quadratic_penalty_method
import numpy as np

def inequality_constraint(x:np.array,feature:np.array,expected_output:np.array) -> tuple:
    return -(np.dot(x[:-1],feature) + x[-1])

def perfect_separation_SVM(features:np.array,expected_outputs:np.array):
    """Will return the optimal normal and bias vectors to define a plane
    to separate the data that can be perfectly separable.

    Parameters
    ----------
    features : np.array
        An list of N dimensional vectors, where there are N features in every vector.
    expected_outputs : np.array
        Expected output for each set of features, must have the same dimension as the vector of feature
        vectors

    Returns
    -------
    tuple
        A vector and a bias
    """

    feature_dimension = len(features[0])
    num_features = len(features)

    if num_features == len(expected_outputs):

        initial_point = np.random.normal(loc = 0, scale = 1,size = feature_dimension + 1)
        objective_function = lambda n : 0.5 * np.linalg.norm(n[:-1]) ** 2 
        inequality_constraints = [
            lambda x : inequality_constraint(x,features[i],expected_outputs[i])
            for i in range(num_features)
        ]
        
        normal_vector = quadratic_penalty_method(
            initial_point,
            12,
            10 ** -6,
            1_000_000,
            2,
            objective_function,
            [],
            inequality_constraints
        )

        norm = normal_vector[:-1]
        bias = normal_vector[-1]
        print(f'The optimal normal vector is: {norm}')
        print(f'The optimal bias is: {bias}')
        return norm,bias
    
    return False
