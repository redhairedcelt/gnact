import random
from collections import defaultdict
from nltk import ngrams
import pandas as pd


def get_uid_history(uid, df_edgelist, print=False):
    """A function to return a df with all the edges for given uid"""
    df = df_edgelist[df_edgelist['uid'] == uid]
    # get all of the previous ports from that uid as sample, except for the last port
    sample = df['Source'].iloc[:].str.replace(' ', '_').values
    # the last port is the target
    target = df['Target'].iloc[-1].replace(' ', '_')
    # concat all the samples into one string
    uid_hist = ''
    for s in sample:
        uid_hist = uid_hist + ' ' + s
    # add the target to the str
    uid_hist = uid_hist + ' ' + target
    if print == True:
        print(f'Previous {str(len(uid_hist.split()) - 1)} ports for {uid} are:', uid_hist.split()[:-1])
        print('Next port is:', target)
    return uid_hist.strip()


def build_history(df_edgelist):
    """
    A function to build a history that includes all port visits per uid as a dict with the uid as
    the key and the strings of each port visited as the values.
    make sure to replace ' ' with '_' in the strings so multi-word ports are one str
    :param df_edgelist:
    :return: a dict with each uid as the key and a space separate set of strings in the order of
    nodes visited.
    """
    history = dict()
    # get all unique uids
    uids = df_edgelist['uid'].unique()
    for uid in uids:
        uid_edgelist = df_edgelist[df_edgelist['uid'] == uid]
        uid_str = ''
        # add all the sources from the source column
        for s in uid_edgelist['Source'].values:
            uid_str = uid_str + ' ' + (s.replace(' ', '_'))
        # after adding all the sources, we still need to add the last target.
        # adding all the sources will provide the history of all but the n-1 port
        uid_str = uid_str + ' ' + (uid_edgelist['Target'].iloc[-1].replace(' ', '_'))
        # only add this history to the dict if the len of the value (# of ports) is >= 2
        if len(uid_str.split()) >= 2:
            history[uid] = uid_str.strip()
    return history


def history_split(history, test_percent=.2):
    """
    A function to split the history dict into a test and train set
    :param history:
    :param test_percent:
    :return: two dictionaries with keys split randomly at the given test percentage.
    """
    history_test = dict()
    history_train = dict()
    for k, v in history.items():
        if random.random() > test_percent:
            history_train[k] = v
        else:
            history_test[k] = v
    return history_train, history_test


def build_ngram_model(history, N):
    """
    A function to that the history dict provided and build N gram model.
    The function will first remove any uids from the history dict that dont have at least
    N number of stops.  Then the function will build a model which is a dict of dicts.
    Each occurence of N-1 ordered stops will be added to the outer dict.  The values of the
    outer dict will an inner dict.  The inner dict's key will be the set of unique Nth
    stops observed and the values will be the probability of that Nth stop being observed
    as a proportion of all Nth stops visited in the corpus.
    :param history:
    :param N:
    :return: a dict of dicts
    """
    # first build a new dict from history that has at least N ports
    historyN = dict()
    for k, v in history.items():
        if len(v.split()) > N:
            historyN[k] = v.strip()
    # Create a placeholder for model that uses the default dict.
    #  the lambda:0 means any new key will have a value of 0
    model = defaultdict(lambda: defaultdict(lambda: 0))
    # build tuple of wN to pass to the model dict
    wordsN = ()
    for i in range(1, N + 1, 1):
        wordsN = wordsN + ('w' + str(i),)
    # Count frequency
    # in history, the key is the uid, the value is the string of ports visited
    for k, v in historyN.items():
        # we split each value and for each Ngram, we populate the model
        # each key is the N-1 ports, and the value is the last port.
        # in this way a trigram uses the first two ports to determine probability
        # the third port was vistied
        for wordsN in ngrams(v.split(), N):
            model[wordsN[:-1]][wordsN[-1]] += 1
    # transform the counts to probabilities and populate the model dict
    for key in model:
        total_count = float(sum(model[key].values()))
        for target in model[key]:
            model[key][target] /= total_count
    return model


def predict_ngram(uid_history, model, N, print=False):
    """
    Given a uid's history, predict the last port visited based on the given model.
    :param uid_history:
    :param model:
    :param N:
    :param print:
    :return: a dict with the predicted next stop given the UID's history, ordered by
    highest probability.
    """
    # check to see if the provided uid history has min N number of stops
    if len(uid_history.split()) < N:
        if print == True:
            print('uid History has fewer than N number of ports visited.')
            print('Cannot make a prediction')
        return None
    else:
        # add the last n ports (except for the last one) to a tuple to pass to the model
        words = ()
        for i in range(N, 1, -1):
            words = words + (uid_history.split()[-i],)
        # get the predicted port based on the model.  predicted is a dict
        predicted = dict(model[words])
        # sort predicted so largest value is first
        predicted = {k: v for k, v in sorted(predicted.items(), key=lambda item: item[1], reverse=True)}

        if print == True:
            print('Top ports (limited to 5) are:')
            # print results
            if len(predicted) >= 5:
                for p in sorted(predicted, key=predicted.get, reverse=True)[:5]:
                    print(p, predicted[p])
            else:
                for p in sorted(predicted, key=predicted.get, reverse=True):
                    print(p, predicted[p])
            # collect results for analysis
            if len(predicted) >= 5:
                for p in (sorted(predicted, key=predicted.get, reverse=True)[:5][0]):
                    if p == uid_history.split()[-1]:
                        print('TRUE!!!')
        return predicted


def evaluate_ngram(uid_history, predicted, top):
    """
    Evaluates if the observed last stop for a UID is in the predicted stops.
    Top is a parameter to control how many ranks will be checked to evaluate prediction.
    :param uid_history:
    :param predicted:
    :param top:
    :return: True or False boolean
    """
    if predicted == None or bool(predicted) == False:
        return None
    else:
        keys = list(predicted.keys())
        target = uid_history.split()[-1]
        if target in keys[:top]:
            return True
        else:
            return False
