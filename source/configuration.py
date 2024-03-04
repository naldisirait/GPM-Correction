class Configuration():
    def __init__(self, config):
        self.config = config

    def get_gpu_index(self):
        return self.config['gpu index']

    def get_path_ann_max_gpm(self):
        return self.config['path_ann_max_gpm']
    
    def get_path_ann_max_stas(self):
        return self.config['path_ann_max_stas']
    
    def get_path_df(self):
        return self.config['path_df']
    
    def get_learning_rate(self):
        return self.config['learning_rate']
    
    def get_skema_test(self):
        return self.config['skema_test']
    
    def get_path_skema_test(self):
        return self.config['path_skema_test']
    
    def get_approach(self):
        return self.config['data processing approach']
    
    def get_batch_size(self):
        return self.config['batch size']
    
    def get_input_size(self):
        return self.config['input size']
    
    def get_hidden_layer(self):
        return self.config['hidden layer']
    
    def get_output_size(self):
        return self.config['output size']
    
    def get_model_name(self):
        return self.config['model name']
    
    def get_ml_model_name(self):
        return self.config["ML model name"]

    def get_epochs(self):
        return self.config['epochs']
    
    def get_T(self):
        return self.config['T']
    
    def get_output_name(self):
        return self.config['output_name']