# from ray.air import session
from copy import deepcopy
import sys
import traceback
from dacite import from_dict

from basic.config import ExecutionConfig, MetaConfig
from h_search_unit import h_search_unit
from basic.exploration_config import ExplorationConfig
import numpy as np
from hs_config_validation_2 import ConfigValidation
from hyperopt import STATUS_FAIL
# def filter_1_conv_result_over_limit(config, max=2000):
#     ae_conv_num = config.get('ae_conv_num', 0)
#     ae_conv_kernel = config.get('ae_conv_kernel', 3)
#     ae_conv_stride = config.get('ae_conv_stride', 1)
#     ae_conv_padding = config.get('ae_conv_padding', 0)
#     def l_out(input_size, kernel, stride, padding, dilation=1):
#         return int((input_size + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#     input_size = 60
#     for i in range(ae_conv_num):
#         input_size = l_out(input_size, ae_conv_kernel, ae_conv_stride, ae_conv_padding)
    
#     if input_size > max:
#         return True
#     return False

def default_objective_function(
    config,
    save_folder,
    dataset_locations,
    basic_experiment_configuration=None,
    exploration_configuration:ExplorationConfig=None,
    additional_info={}):
    basic_experiment_config = deepcopy(basic_experiment_configuration)
    
    if exploration_configuration.validation is not None:
        for validation in exploration_configuration.validation:
            # print('Validation:', validation)
            validation_object = ConfigValidation(validation)
            validation_result = validation_object.validate(config)
            if validation_result is None:
                raise ValueError('Validation result is None.')
            if not validation_result:
                return {
                    'status': STATUS_FAIL,
                    'score': float('-inf'),
                    'num_params': -1,
                    'num_trainable_params': -1,
                    'error_type': 'Config not permitted',
                    'error_message': f'Validation {validation.validation_type} + {validation.validation_subtype} failed. Returning {validation.exception_value}.',
                    'error_traceback': 'Check the validation configuration.'
                }

    # Check for the limit
    # if filter_1_conv_result_over_limit(config):
        
    for search_space_unit in exploration_configuration.search_space:
        # property_content = config[key] if key in config else []
        property_content = config[search_space_unit.identifier] if search_space_unit.identifier in config else []
        # if search_space[key]['tune_function'] == 'multichoice':
        if search_space_unit.tune_function == 'multichoice':
            # Get the values
            property_content = []
            # for parameter in search_space[key]['tune_parameters']:
            for parameter in search_space_unit.tune_parameters:
                multichoice_key = f"MC-{search_space_unit.identifier}-{parameter}"
                if config[multichoice_key] == 1:
                    property_content.append(parameter)
            if property_content == []:
                return {
                    'status': STATUS_FAIL,
                    'score': float('-inf'),
                    'num_params': -1,
                    'num_trainable_params': -1,
                    'error_type': '0 reducer',
                    'error_message': 'There is no reducer in the configuration.',
                    'error_traceback': 'There is no reducer in the configuration.'
                }
        if search_space_unit.tune_function == 'quniform' and search_space_unit.tune_parameters[-1] == 1:
            property_content = int(property_content)
        # Prepare the route
        route = search_space_unit.route.split('/')
        property_to_modify = basic_experiment_config
        for key, item in enumerate(route[:-1]):
            # If item is a number, then it is a list
            if item.isdigit():
                item = int(item)
            property_to_modify = property_to_modify[item]
        # Set the value
        property_to_modify[route[-1]] = property_content
    config_to_execute = from_dict(
        data_class=ExecutionConfig,
        data=basic_experiment_config
    )
    if config_to_execute.metadata is None:
        config_to_execute.metadata = from_dict(
            data_class=MetaConfig,
            data={'experiment_type': 'default'}
        ) 
    try:
        result = h_search_unit(
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            config_to_execute=config_to_execute,
            additional_info=additional_info
        )
    except Exception as e:
        print('EXCEPTION FOUND\n', e)
        syserror = sys.exc_info()
        result = {
            'score': -0.001,
            'num_params': -1,
            'num_trainable_params': -1,
            'error_type': str(syserror[0]),
            'error_message': str(syserror[1]),
            'error_traceback': '\n'.join(traceback.format_tb(e.__traceback__))
        }
    return result

# def new_objective_function(
#     config,
#     save_folder,
#     dataset_locations,
#     basic_experiment_configuration=None,
#     exploration_configuration=None):
#     basic_experiment_config = deepcopy(basic_experiment_configuration)
#     # Update the values for the current experiment
#     search_space = exploration_configuration['search_space']
#     for key in search_space:
#         property_content = config[key] if key in config else []
#         if search_space[key]['tune_function'] == 'multichoice':
#             # Get the values
#             property_content = []
#             for parameter in search_space[key]['tune_parameters']:
#                 multichoice_key = f"MC-{key}-{parameter}"
#                 if config[multichoice_key] == 1:
#                     property_content.append(parameter) 
#         # Prepare the route
#         route = search_space[key]['route'].split('/')
#         property_to_modify = basic_experiment_config
#         for key, item in enumerate(route[:-1]):
#             # If item is a number, then it is a list
#             if item.isdigit():
#                 item = int(item)
#             property_to_modify = property_to_modify[item]
#         # Set the value
#         property_to_modify[route[-1]] = property_content
    
#     config_to_execute = from_dict(
#         data_class=ExecutionConfig,
#         data=basic_experiment_config
#     )
#     try:
#         result = h_search_unit(
#             save_folder=save_folder,
#             dataset_locations=dataset_locations,
#             config_to_execute=config_to_execute,
#             experiment_type='custom',
#             extra_info=exploration_configuration['additional_info']
#         )
#     except Exception as e:
#         print('EXCEPTION FOUND\n', e)
#         syserror = sys.exc_info()
#         result = {
#             'score': -0.001,
#             'num_params': -1,
#             'num_trainable_params': -1,
#             'error_type': str(syserror[0]),
#             'error_message': str(syserror[1]),
#             'error_traceback': '\n'.join(traceback.format_tb(e.__traceback__))
#         }
#     #     scores.append(result['score']/float(exploration_configuration['additional_info'][dataset]))
#     #     result_compilation.update({f'{dataset}-{key}': result[key] for key in result})
#     # result_compilation['score'] = sum(scores)/len(scores) # MEAN
#     session.report(result)
        
         