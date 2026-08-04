from src.method_functions.LSB_methods_functions import LSB_method
from src.method_functions.RLSB_methods_functions import RLSB_method


def estimate_eecm_parameters(method_input,method_group,method_type,method_solver):
    """
    Docstring for estimate_eecm_parameters
    
    :param method_input: Description
    :param method_group: Description
    :param method_type: Description

    :return estimated_parameter: array with theta values
    """

    if method_group == "LSB":

        parameter_estimation_fn = LSB_method.get(method_group+"_"+method_type+"_method")

        estimated_parameter = parameter_estimation_fn(method_input[method_group][method_type],method_solver) # the output is an array

    elif method_group == "RLSB":

        parameter_estimation_fn = RLSB_method.get(method_group+"_"+method_type+"_method")

        estimated_parameter = parameter_estimation_fn(method_input[method_group][method_type],method_solver) # the output is an array


    

    return estimated_parameter