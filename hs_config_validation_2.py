from basic.exploration_config import ValidationConfig

class ConfigValidation:
    def __init__(self, val_config: ValidationConfig):
        self.val_config = val_config
    
    def validate(self, config):
        function_name = f'validator_for_type_{self.val_config.validation_type}'
        if hasattr(self, function_name) and callable(function_to_call := getattr(self, function_name)):
            return function_to_call(config)
        return None

    def validator_for_type_comparative(self, config):
        if self.val_config.validation_subtype == 'less_than':
            return config[self.val_config.validation_parameters[0]] < config[self.val_config.validation_parameters[1]]
        elif self.val_config.validation_subtype == 'greater_than':
            return config[self.val_config.validation_parameters[0]] > config[self.val_config.validation_parameters[1]]
        elif self.val_config.validation_subtype == 'equal':
            return config[self.val_config.validation_parameters[0]] == config[self.val_config.validation_parameters[1]]
        elif self.val_config.validation_subtype == 'not_equal':
            return config[self.val_config.validation_parameters[0]] != config[self.val_config.validation_parameters[1]]
        elif self.val_config.validation_subtype == 'less_than_or_equal':
            return config[self.val_config.validation_parameters[0]] <= config[self.val_config.validation_parameters[1]]
        elif self.val_config.validation_subtype == 'greater_than_or_equal':
            return config[self.val_config.validation_parameters[0]] >= config[self.val_config.validation_parameters[1]]
        else:
            return False


    def validate_last_conv_layer_neurons(self, config, max=2000):
        ae_conv_num = int(config.get('ae_conv_num', 0))
        ae_conv_kernel = int(config.get('ae_conv_kernel', 3))
        ae_conv_stride = int(config.get('ae_conv_stride', 1))
        ae_conv_padding = int(config.get('ae_conv_padding', 0))
        def l_out(input_size, kernel, stride, padding, dilation=1):
            return int((input_size + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        input_size = 60
        for i in range(ae_conv_num):
            input_size = l_out(input_size, ae_conv_kernel, ae_conv_stride, ae_conv_padding)
        
        if input_size*256 < max:
            return True
        return False

    # def validate_simulation_for_convae_size_smaller_than(self, config):
    #     # First parameter: limit value
    #     limit_value = self.val_config.validation_parameters[0]
    #     config['ae_conv_num'] = 3
    #     config['ae_conv_kernel'] = 3
    #     config['ae_conv_stride'] = 1
    #     config['ae_conv_padding'] = 0
    #     def l_out(l_in, kernel, stride, padding, dilation=1):
    #         return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)

    def validator_for_type_last_conv_layer_evaluation(self, config):
        subtype = self.val_config.validation_subtype
        if subtype == 'less_than':
            return self.validate_last_conv_layer_neurons(config, self.val_config.validation_parameters[0])
        return False

        # # First parameter: model type
        # model_type = config[self.val_config.validation_parameters[0]] # Possible values: 'convae'
        # # Second parameter: limit type
        # limit_type = self.val_config.validation_parameters[1] # Possible values: 'less_than', 'greater_than', 'equal', 'not_equal', 'less_than_or_equal', 'greater_than_or_equal'
        # if model_type == 'convae':
        #     return True
        # return False