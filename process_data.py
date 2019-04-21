import csv
import pickle
import numpy
import spacy

from keras_preprocessing.sequence import pad_sequences
from src.application import Application
from src.file_util import write_file

nlp = spacy.load('en')


def translate(tokenizer_data):
    emb = read_emb()
    word2tokenizer = {'<unk>': 0}
    filtered_emb = [numpy.random.uniform(-0.1, 0.1, 300)]
    for data in tokenizer_data:
        for i in range(len(data['q1'])):
            translate_tokenizer(data['q1'][i], word2tokenizer, emb, filtered_emb)
            translate_tokenizer(data['q2'][i], word2tokenizer, emb, filtered_emb)
            if len(data['q1'][i]) > Application.model_params['max_sequence_length']:
                data['q1'][i] = data['q1'][i][:Application.model_params['max_sequence_length']]
            if len(data['q2'][i]) > Application.model_params['max_sequence_length']:
                data['q2'][i] = data['q2'][i][:Application.model_params['max_sequence_length']]
        data['q1'] = pad_sequences(data['q1'], maxlen=Application.model_params['max_sequence_length'], padding='post')
        data['q2'] = pad_sequences(data['q2'], maxlen=Application.model_params['max_sequence_length'], padding='post')
    filtered_emb = numpy.asarray(filtered_emb, dtype='float32')
    return filtered_emb, word2tokenizer


def translate_tokenizer(tokens, token_map, emb, weight):
    for i in range(len(tokens)):
        if tokens[i] not in token_map:
            if tokens[i] in emb:
                token_map[tokens[i]] = len(token_map)
                weight.append(emb[tokens[i]])
            else:
                tokens[i] = '<unk>'
        tokens[i] = token_map[tokens[i]]


def get_tokenizer_data(file_data):
    tokenizer_data = []
    for i in range(len(file_data)):
        item = {'q1': [], 'q1_length': [], 'q2': [], 'q2_length': [], 'y': [], 'q1_text': [], 'q2_text': []}
        for j in range(len(file_data[i])):
            q1 = participle_row_data(file_data[i][j]['question1'])
            q2 = participle_row_data(file_data[i][j]['question2'])
            item['q1'].append(q1)
            item['q2'].append(q2)
            item['q1_length'].append(len(q1))
            item['q2_length'].append(len(q2))
            item['y'].append(int(file_data[i][j]['is_duplicate']))
            if i == 2:
                item['q1_text'].append(participle_row_data(file_data[i][j]['question1']))
                item['q2_text'].append(participle_row_data(file_data[i][j]['question2']))
        item['q1_length'] = numpy.asarray(item['q1_length'], dtype='int32')
        item['q2_length'] = numpy.asarray(item['q2_length'], dtype='int32')
        tokenizer_data.append(item)
    return tokenizer_data


def save_new_dataset(tokenizer_data):
    words = []
    for data in tokenizer_data:
        for q1, q2 in zip(data['q1'], data['q2']):
            words.append(" ".join(q1))
            words.append(" ".join(q2))
    write_file("data/quora_duplicate_questions_words.txt", words)


def participle_row_data(data):
    if data is None:
        return []
    text = data.strip()
    doc = nlp.tokenizer(text)
    tokens = [t.lower_ for t in doc]
    return tokens


def read_file_data(file=Application.data['data_file']):
    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = list(reader)
        data = numpy.asarray(data)
        numpy.random.seed(666)
        numpy.random.shuffle(data)
        length = data.shape[0]
        train = data[:int(0.8 * length)]
        valid = data[int(0.8 * length):int(0.9 * length)]
        test = data[int(0.9 * length):]
        return train, valid, test


def read_emb(file=Application.data['emb_file']):
    emb = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if len(tokens) == 301:
                emb[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
    return emb


def process():
    print('Reading file...')
    file_data = read_file_data()
    print('Read file done.\nGetting tokenizer data...')
    tokenizer_data = get_tokenizer_data(file_data)
    # save_new_dataset(tokenizer_data)
    print('Get tokenizer data done.\nTranslating data...')
    emb_matrix, word2tokenizer = translate(tokenizer_data)
    print('Translate data done.')
    with open(Application.model['app_data'], 'wb') as f:
        pickle.dump((tokenizer_data, emb_matrix, word2tokenizer), f)
    print('Saved.')
    return tokenizer_data, emb_matrix, word2tokenizer


if __name__ == '__main__':
    process()
