from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import multiply, concatenate


from keras.layers import Dense, Input, LSTM, Embedding, add, Conv1D, \
    GlobalMaxPooling1D, Dropout, GlobalAveragePooling1D, subtract, GRU, Activation
from keras.optimizers import Adam

from src.application import Application
from src.attention import Attention, MultiHeadAttention
from src.merges import SubtractAbs, AddMean


class NeuralNetworksModels(object):
    def __init__(self, emb_matrix, model_style):
        self.emb_matrix = emb_matrix
        self.model_style = model_style

    def model(self, file=None):
        sequence_1_input, sequence_2_input, sequence_3_input, sequence_4_input, embedded_sequences_1, embedded_sequences_2 = self.get_input()
        # 所有神经网络模型
        predictions = self.get_model(embedded_sequences_1, embedded_sequences_2, sequence_3_input, sequence_4_input)
        # 多层感知机
        predictions = self.mlp(predictions)
        return compile_model([sequence_1_input, sequence_2_input, sequence_3_input, sequence_4_input], predictions,
                             file=file)

    def get_input(self):
        sequence_1_input = Input(shape=(Application.model_params['max_sequence_length'],), dtype='float64')
        sequence_2_input = Input(shape=(Application.model_params['max_sequence_length'],), dtype='float64')
        sequence_3_input = Input(shape=(1,), dtype='int32')
        sequence_4_input = Input(shape=(1,), dtype='int32')
        embedding_layer = Embedding(self.emb_matrix.shape[0], self.emb_matrix.shape[1], weights=[self.emb_matrix],
                                    input_length=Application.model_params['max_sequence_length'],
                                    trainable=False)
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        return sequence_1_input, sequence_2_input, sequence_3_input, sequence_4_input, embedded_sequences_1, embedded_sequences_2

    def get_model(self, embedded_sequences_1, embedded_sequences_2, sequences_1_length, sequences_2_length):
        if self.model_style == 'bi_lstm':
            print('using model bi_lstm!!!')
            model_layer1 = Bidirectional(LSTM(Application.model_params['num_nn']))
        elif self.model_style == 'ap_bi_lstm':
            print('using model ap_bi_lstm!!!')
            model_layer1 = Bidirectional(LSTM(Application.model_params['num_nn'], return_sequences=True))
            x_1 = model_layer1(embedded_sequences_1)
            x_1 = Attention()(x_1)
            y_1 = model_layer1(embedded_sequences_2)
            y_1 = Attention()(y_1)
            return concatenate([x_1, y_1, SubtractAbs()([x_1, y_1]), multiply([x_1, y_1])])
        elif self.model_style == 'bi_gru':
            print('using model bi_gru!!!')
            model_layer1 = Bidirectional(GRU(Application.model_params['num_nn']))
        elif self.model_style == 'ap_bi_gru':
            print('using model ap_bi_gru!!!')
            model_layer1 = Bidirectional(GRU(Application.model_params['num_nn'], return_sequences=True))
            x_1 = model_layer1(embedded_sequences_1)
            x_1 = Attention()(x_1)
            y_1 = model_layer1(embedded_sequences_2)
            y_1 = Attention()(y_1)
            return concatenate([x_1, y_1, SubtractAbs()([x_1, y_1]), multiply([x_1, y_1])])
        elif self.model_style == 'cnn':
            model_layer1 = Conv1D(Application.model_params['num_nn'], 2,
                                  padding='valid', activation='relu', strides=1)
            x_1 = model_layer1(embedded_sequences_1)
            y_1 = model_layer1(embedded_sequences_2)

            x_1 = GlobalMaxPooling1D()(x_1)
            y_1 = GlobalMaxPooling1D()(y_1)

            return concatenate([x_1, y_1, SubtractAbs()([x_1, y_1]), multiply([x_1, y_1])])
        elif self.model_style == 'ap_cnn':
            model_layer1 = Conv1D(Application.model_params['num_nn'], 2,
                                  padding='valid', activation='relu', strides=1)
            x_1 = model_layer1(embedded_sequences_1)
            y_1 = model_layer1(embedded_sequences_2)
            x_1 = Attention()(x_1)
            y_1 = Attention()(y_1)
            return concatenate([x_1, y_1, SubtractAbs()([x_1, y_1]), multiply([x_1, y_1])])
        elif self.model_style == 'multi_attention':
            print('using model multi_attention!!!')
            x_1_1 = Dense(Application.model_params['num_nn'])(embedded_sequences_1)
            y_1_1 = Dense(Application.model_params['num_nn'])(embedded_sequences_2)

            x_2, y_2 = multi_head_self_attention(x_1_1, y_1_1)
            x_3, y_3 = multi_head_mutual_attention(x_1_1, y_1_1)

            z_2 = concatenate([x_2, y_2], axis=2)
            z_2 = GlobalMaxPooling1D()(z_2)
            z_3 = concatenate([x_3, y_3], axis=2)
            z_3 = GlobalMaxPooling1D()(z_3)
            return concatenate([z_2, z_3])
        else:
            print("did not find this style model")
        x_1 = model_layer1(embedded_sequences_1)
        y_1 = model_layer1(embedded_sequences_2)
        return concatenate([SubtractAbs()([x_1, y_1]), multiply([x_1, y_1])])

    def mlp(self, merged):
        if self.model_style == 'multi_attention':
            merged = Dense(2048, activation='relu')(merged)
            merged = Dense(64)(merged)
            merged = Dense(8)(merged)
            merged = Dense(4)(merged)
        else:
            merged = Dense(2048)(merged)
            merged = Dense(32)(merged)
            merged = Dense(8)(merged)
        return Dense(1, activation='sigmoid')(merged)


def compile_model(inputs, predictions, file=None):
    model = Model(inputs=inputs, outputs=predictions)
    if file is not None:
        model.load_weights(file)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=Application.model_params['lr']),
                  metrics=['accuracy'])
    model.summary()
    return model


def multi_head_mutual_attention(x, y, x_l=None, y_l=None):
    z_1 = multi_attention(x, y, y, x_l, y_l)
    z_2 = multi_attention(y, x, x, y_l, x_l)
    return add_forward(x, z_1), add_forward(y, z_2)


def multi_head_self_attention(x, y, x_l=None, y_l=None):
    z_1 = multi_attention(x, x, x, x_l, x_l)
    z_2 = multi_attention(y, y, y, y_l, y_l)
    return add_forward(x, z_1), add_forward(y, z_2)


def multi_attention(q, k, v, q_l=None, k_l=None):
    if q_l is not None:
        x = MultiHeadAttention(Application.model_params['head'],
                               int(Application.model_params['num_nn'] / Application.model_params['head']))(
            [q, k, v, q_l, k_l])
    else:
        x = MultiHeadAttention(Application.model_params['head'],
                               int(Application.model_params['num_nn'] / Application.model_params['head']))([q, k, v])
    return x


def add_forward(x, y):
    z = add([x, y])
    z_1 = Dense(int(Application.model_params['num_nn'] * 2), activation='relu')(z)
    z_1 = Dense(Application.model_params['num_nn'])(z_1)
    return z_1
