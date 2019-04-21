from src.application import Application


def write_result_file(test_set, predicts, hist, score, model_style, acc):
    write_texts = []
    for j in range(len(test_set['y'])):
        write_texts.append("%.4g\t %s t1: %s\t t2: %s" %
                           (predicts[j], test_set['y'][j], " ".join(test_set['q1_text'][j]),
                            " ".join(test_set['q2_text'][j])))
    write_texts.append("test acc:%.4g\t test score:%s\t history acc:%s\t history score:%s" % (
        acc, score, max(hist.history['acc']), min(hist.history['loss'])))
    print("test acc:%s\t test score:%s\t history acc:%s\t history score:%s" % (
        acc, score, max(hist.history['acc']), min(hist.history['loss'])))
    file = Application.directory['data'] + Application.model_params['system'] + '_' + str(
        Application.model_params['epochs']) + '_' + model_style + '_acc_' + str(acc) + '.csv'
    write_file(file, write_texts)


def write_file(file, texts):
    print("write text in file " + file)
    with open(file, encoding='utf-8', mode='w') as f:
        for line in texts:
            f.write(line + '\n')
