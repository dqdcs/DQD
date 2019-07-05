import os, sys
import pickle

import tensorflow as tf
from keras.callbacks import ModelCheckpoint

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# tensorflow按需求申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from src.application import Application
from src.neural_networks_models import NeuralNetworksModels
from src.file_util import write_result_file


def run():
    print('Loading data...')
    app = Application()
    with open(Application.model['app_data'], 'rb') as f:
        tokenizer_data, emb_matrix, word2tokenizer = pickle.load(f)
    train_set = tokenizer_data[0]
    dev_set = tokenizer_data[1]
    test_set = tokenizer_data[2]
    # style_models = ['bi_gru_multi_attention', 'multi_attention', 'bi_lstm', 'ap_bi_lstm', 'ap_bi_gru', 'bi_gru', 'cnn',
    #                 'ap_cnn']
    style_models = ['bi_gru_multi_attention']
    for i in range(100):
        for style_model in style_models:
            train = [train_set['q1'], train_set['q2'], train_set['q1_length'], train_set['q2_length']]
            dev = [dev_set['q1'], dev_set['q2'], dev_set['q1_length'], dev_set['q2_length']]
            test = [test_set['q1'], test_set['q2'], test_set['q1_length'], test_set['q2_length']]
            model = NeuralNetworksModels(emb_matrix, style_model).model()
            file_path = Application.directory['data'] + 'model-train-normal-' + style_model + '.h5'
            checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1,
                                         save_weights_only=True)
            hist = model.fit(train, train_set['y'], callbacks=[checkpoint],
                             validation_data=[dev, dev_set['y']],
                             epochs=app.model_params['epochs'],
                             batch_size=app.model_params['batch_size'])
            score_1, acc_1 = model.evaluate(x=test, y=test_set['y'], batch_size=app.model_params['batch_size'])
            predicts_1 = model.predict(test, batch_size=app.model_params['batch_size'])
            model = NeuralNetworksModels(emb_matrix, style_model).model(file_path)
            score, acc = model.evaluate(x=test, y=test_set['y'], batch_size=app.model_params['batch_size'])
            predicts = model.predict(test, batch_size=app.model_params['batch_size'])
            if acc_1 > acc:
                score = score_1
                acc = acc_1
                predicts = predicts_1
            print("current best acc:%s\t test score:%s" % (acc, score))
            if acc > app.learner[style_model]:
                app.learner[style_model] = acc
                print("save acc:%s\t test score:%s" % (acc_1, score_1))
                with open(Application.directory['model'] + style_model + Application.model['predict'],
                          'wb') as f:
                    pickle.dump(predicts, f)
                write_result_file(test_set, predicts, hist, score, style_model, acc)


if __name__ == '__main__':
    run()
