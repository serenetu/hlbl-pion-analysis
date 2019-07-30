import numpy as np


def make_jackknife_dict(dict):

    keys = dict.keys()
    jackknife_dic = {}
    for i in range(len(keys)):
        array = []
        new_configs = keys[:i] + keys[i+1:]

        for config in new_configs:
            array.append(dict[config])
        array = np.array(array)
        mean = np.mean(array, axis=0)
        jackknife_dic[keys[i]] = mean

    return jackknife_dic


def jackknife_avg_err_dict(jackknife_dict):
    mean, err = jackknife_avg_err_array(jackknife_dict.values())
    return mean, err


def jackknife_avg_err_array(jackknife_array):
    array = np.array(jackknife_array)
    mean = np.mean(array, axis=0)
    err = (len(jackknife_array) - 1.) ** (1. / 2.) * np.std(array, axis=0)
    return mean, err
