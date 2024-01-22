class Configuration():
    def __init__(self, config):
        self.config = config

    def get_path_train_data(self):
        return self.config['path_train_dataset']
    
    def get_path_test_data(self):
        return self.config['path_test_dataset']
    
    def get_features_name(self):
        return self.config['features']
    
    def get_target_name(self):
        return self.config['target']
    
    def get_criterion(self):
        pass
    def get_learning_rate(self):
        pass
    def get_optimizer(self):
        pass