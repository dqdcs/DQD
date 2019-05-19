import pickle, numpy
from sklearn.metrics import accuracy_score
import multiprocessing

from src.application import Application


def _evaluation(pref, truth):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for p, t in zip(pref, truth):
        if t == 1:
            if p == 1:
                TP += 1
            else:
                FP += 1
        else:
            if p == 0:
                TN += 1
            else:
                FN += 1
    P = TP / (TP + FP) * 100
    R = TP / (TP + FN) * 100
    F1 = 2 * (P * R) / (P + R)
    return P, R, F1


def _credible(row, alpha):
    one = numpy.zeros_like(row)
    zero = numpy.zeros_like(row)
    one[row > 0.5 + alpha] = 1
    zero[row < 0.5 - alpha] = 1
    one = int(numpy.sum(one))
    zero = int(numpy.sum(zero))
    return zero, one


def _vote(row, alpha, limit):
    zero, one = _credible(row, alpha)
    y = numpy.argmax([zero, one, limit + 1])
    if y != 2:
        return zero, one, y
    else:
        close_0 = min(row)
        close_1 = 1.0 - max(row)
        return zero, one, numpy.argmin([close_0, close_1])


def _credible_voting(truths, truth, alpha):
    pref = []
    vote = [0, 0, 0, 0, 0, 0, 0, 0]
    truths = numpy.asarray(truths).transpose((1, 0, 2))
    limit = int(truths.shape[1] / 2)
    for row, label in zip(truths, truth):
        zero, one, p = _vote(row, alpha, limit)
        pref.append(p)
        if label == 1:
            vote[one] += 1
        else:
            vote[zero] += 1
    pref = numpy.asarray(pref, dtype='int64')
    return pref, accuracy_score(pref, truth), alpha, vote


def grid_search_credible_voting(truths, truth, show=False):
    best_acc = 0
    best_alpha = -1
    best_pref = []
    result = []
    pool = multiprocessing.Pool(processes=12)
    vote = None
    for alpha in range(11):
        alpha = alpha / 20
        result.append(pool.apply_async(_credible_voting, (truths, truth, alpha)))
    pool.close()
    pool.join()
    result = [res.get() for res in result]
    if show:
        print("best_acc:" + str(best_acc) + "   best_alpha:" + str(best_alpha))
        for v in result[0][3]:
            print(v)
    for res in result:
        if best_acc < res[1]:
            [best_pref, best_acc, best_alpha, vote] = res
        if show:
            print(res[1])
    P, R, F1 = _evaluation(best_pref, truth)
    if show:
        print("best_acc:" + str(best_acc) + "   best_alpha:" + str(best_alpha))
        for v in vote:
            print(v)
    return best_pref, best_acc, best_alpha, P, R, F1


def max_credible_voting(truths, truth):
    pref = numpy.zeros([truths[0].shape[0], 2])
    for i in range(len(truths)):
        pref = numpy.array(list(
            map(lambda p, x: [numpy.max([p[0], 1 - x[0]]), numpy.max([p[1], x[0]])], pref, truths[i])))
    pref = numpy.argmax(pref, axis=-1)
    P, R, F1 = _evaluation(pref, truth)
    acc = accuracy_score(pref, truth)
    return pref, acc, P, R, F1


def voting(truths, truth):
    pref = numpy.zeros([truths[0].shape[0], 2])
    for i in range(len(truths)):
        pref = numpy.sum([pref, numpy.array(list(map(lambda x: [0, 1] if x >= 0.5 else [1, 0], truths[i])))], axis=0)
    pref = numpy.argmax(pref, axis=-1)
    P, R, F1 = _evaluation(pref, truth)
    acc = accuracy_score(pref, truth)
    return pref, acc, P, R, F1


def average(truths, truth):
    pref = numpy.zeros([truths[0].shape[0], 2])
    for i in range(len(truths)):
        pref = numpy.array(list(
            map(lambda p, x: [numpy.sum([p[0], 1 - x[0]]), numpy.sum([p[1], x[0]])], pref, truths[i])))
    pref = numpy.argmax(pref, axis=-1)
    acc = accuracy_score(pref, truth)
    return pref, acc


def ensemble(styles, show=False):
    truths = []
    with open(Application.model['app_data'], 'rb') as f:
        tokenizer_data, emb_matrix, word2tokenizer = pickle.load(f)
    truth = tokenizer_data[2]['y']
    for i in range(len(styles)):
        print(styles[i], end='\t')
        with open(Application.directory['model'] + styles[i] + Application.model['predict'], 'rb') as f:
            truths.append(pickle.load(f))
    print()
    result = []
    _, acc, _, P, R, F1 = grid_search_credible_voting(truths, truth, show)
    pool = multiprocessing.Pool(processes=3)
    result.append(pool.apply_async(voting, (truths, truth)))
    result.append(pool.apply_async(max_credible_voting, (truths, truth)))
    result.append(pool.apply_async(average, (truths, truth)))
    pool.close()
    pool.join()
    result = [res.get() for res in result]
    # print('Voting:%.2f & %.2f & %.2f & %.2f' % (
    #     result[0][1] * 100, result[0][2], result[0][3], result[0][4]))
    # print('MCV:%.2f(+%.2f) & %.2f & %.2f & %.2f' % (
    #     result[1][1] * 100, result[1][1] * 100 - result[0][1] * 100, result[1][2],
    #     result[1][3], result[1][4]))
    # print('CV:%.2f(+%.2f) & %.2f & %.2f & %.2f' % (acc * 100, acc * 100 - result[0][1] * 100, P, R, F1))
    # print('Ave:%.2f(+%.2f)' % (result[2][1] * 100, result[2][1] * 100 - acc * 100))
    print('paper:%.1f & %.1f(+%.1f) & %.1f(+%.1f)' % (
        result[0][1] * 100, result[1][1] * 100, result[1][1] * 100 - result[0][1] * 100, acc * 100,
        acc * 100 - result[0][1] * 100))


def statistic():
    with open(Application.model['app_data'], 'rb') as f:
        tokenizer_data, emb_matrix, word2tokenizer = pickle.load(f)
    truth = tokenizer_data[2]['y']
    for style in ['bi_lstm', 'ap_bi_lstm', 'bi_gru', 'ap_bi_gru', 'cnn', 'ap_cnn', 'multi_attention']:
        with open(Application.directory['model'] + style + Application.model['predict'], 'rb') as f:
            pref = numpy.array(list(map(lambda x: 1 if x >= 0.5 else 0, pickle.load(f))))
        P, R, F1 = _evaluation(pref, truth)
        Acc = accuracy_score(pref, truth) * 100
        # print('%-17s:%.2f & %.2f & %.2f & %.2f' % (style, Acc, P, R, F1))
        print('%-17s:%.1f & %.1f & %.1f & %.1f' % (style, Acc, P, R, F1))


if __name__ == '__main__':
    statistic()
    ensemble(['multi_attention', 'bi_lstm'])
    ensemble(['multi_attention', 'ap_bi_lstm'])
    ensemble(['multi_attention', 'bi_gru'])
    ensemble(['multi_attention', 'ap_bi_gru'])
    ensemble(['multi_attention', 'cnn'])
    ensemble(['multi_attention', 'ap_cnn'])

    ensemble(['multi_attention', 'bi_lstm', 'ap_bi_lstm'])
    ensemble(['multi_attention', 'bi_gru', 'ap_bi_gru'])
    ensemble(['multi_attention', 'cnn', 'ap_cnn'])

    ensemble(['multi_attention', 'bi_lstm', 'cnn'])
    ensemble(['multi_attention', 'bi_lstm', 'bi_gru'])
    ensemble(['multi_attention', 'bi_gru', 'cnn'])

    ensemble(['multi_attention', 'ap_bi_lstm', 'ap_bi_gru'])
    ensemble(['multi_attention', 'ap_bi_lstm', 'ap_cnn'])
    ensemble(['multi_attention', 'ap_cnn', 'ap_bi_gru'])

    ensemble(['multi_attention', 'bi_lstm', 'ap_bi_lstm', 'ap_bi_gru', 'bi_gru'])
    ensemble(['multi_attention', 'bi_lstm', 'ap_bi_lstm', 'cnn', 'ap_cnn'])
    ensemble(['multi_attention', 'cnn', 'ap_cnn', 'ap_bi_gru', 'bi_gru'])

    ensemble(['multi_attention', 'bi_lstm', 'ap_bi_lstm', 'ap_bi_gru', 'bi_gru', 'cnn', 'ap_cnn'], show=True)
