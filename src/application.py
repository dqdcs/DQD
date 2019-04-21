import platform


class Application(object):
    directory = {'data': './data/', 'model': './model/', }
    model = {'app_data': directory['model'] + 'test.data', 'predict': '.predict',
             'all_data': directory['model'] + 'app.data'}
    data = {'emb_file': directory['data'] + 'glove.840B.300d.txt',
            'data_file': directory['data'] + 'quora_duplicate_questions.tsv'}
    model_params = {'epochs': 3, 'batch_size': 256, 'max_sequence_length': 64, 'lr': 0.001,
                    'system': 'Linux', 'num_nn': 256, 'num_dense': 300, 'head': 8}
    learner = {'multi_attention': 0.8, 'bi_lstm': 0.8, 'ap_bi_lstm': 0.8, 'ap_bi_gru': 0.8, 'bi_gru': 0.8, 'cnn': 0.8,
               'ap_cnn': 0.8}

    def __init__(self):
        system = platform.system()
        if system == "Linux":
            self.model_params['batch_size'] = 512
            self.model_params['epochs'] = 30
            self.model['app_data'] = self.directory['model'] + 'app.data'
        else:
            self.model_params['system'] = system
            self.model['app_data'] = self.directory['model'] + 'app.data'