import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from ensemble import *
from extrapolation import *
import jackknife as jk


def get_fr_path(ensemble):
    path = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/fr/' + ensemble + '/ama'
    return path


def get_fr_model_path(ensemble):
    path = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/fr/' + ensemble + '/model/model'
    return path


def read_complex_bi(path):
    x = np.fromfile(path, dtype='complex128')
    return x


def read_all_traj(path):
    res = []
    files = os.listdir(path)
    for file in files:
        if 'results' not in file:
            continue
        if not os.path.isfile(path + '/' + file):
            continue
        one = read_complex_bi(path + '/' + file)
        res.append(one)
    return np.array(res)


def get_x_axis(length):
    x = []
    for r2 in range(length):
        x.append(r2 ** (1. / 2.))
    return np.array(x)


def rm_fr_0(x, fr, err=None):
    i_to_rm = []
    for i in range(len(fr)):
        if fr[i] == 0.:
            i_to_rm.append(i)
    fr_ = np.delete(fr, i_to_rm)
    x_ = np.delete(x, i_to_rm)
    if err is not None:
        err_ = np.delete(err, i_to_rm)
        return x_, fr_, err_
    return x_, fr_


def plt_one_fr(fr, unit):
    length = len(fr)
    x = get_x_axis(length) * unit
    plt.plot(x, fr, '.')
    return


def plt_fr_mean_err(mean, unit, err=None, label=''):
    length = len(mean)
    x = get_x_axis(length) * unit
    if err is None:
        x_, mean_ = rm_fr_0(x, mean, err)
        plt.errorbar(x_, mean_, fmt='*', label=label)
    else:
        x_, mean_, err_ = rm_fr_0(x, mean, err)
        plt.errorbar(x_, mean_, yerr=err_, fmt='*', label=label)

    return


def get_x_interpolation_axis(length, r_max):
    r_step = r_max / (length - 1)  # in fm
    x = [i * r_step for i in range(length)]
    return np.array(x)


def plt_fr_interpolation_mean_err(mean, r_max, err=None, label='', shape='*', color='r'):
    length = len(mean)
    x = get_x_interpolation_axis(length, r_max)
    if err is not None:
        plt.errorbar(x, mean, fmt=shape, label=label, color=color)
    else:
        plt.errorbar(x, mean, yerr=err, fmt=shape, label=label, color=color)
    return


def get_fr_all_traj(ensemble):
    path = get_fr_path(ensemble)
    fr = read_all_traj(path)
    if 'interpolation' in ensemble:
        ensemble_ = '-'.join(ensemble.split('-')[:-1])
    else:
        ensemble_ = ensemble
    # fac = compute_factor(ensemble_)
    # fr = fac * fr
    return fr


def get_fr_mean_err(fr):
    mean = np.mean(fr, axis=0)
    err = 1. / len(fr) ** (1. / 2.) * np.std(fr, axis=0, ddof=1)
    # err = np.std(fr, axis=0, ddof=1)
    return mean, err


def get_fr_integration_one_traj(fr, ensemble, r_max, num_p=1000):
    length = len(fr)
    x = get_x_interpolation_axis(length, r_max)
    xvals = np.linspace(0, r_max, num_p)
    yinterp = np.interp(xvals, x, fr)
    dr = 1. * r_max / num_p / get_a(ensemble)  # lattice spacing
    intergration = [0.]
    for i in range(1, len(xvals)):
        # intergration.append(intergration[-1] + yinterp[i] * (xvals[i] / get_a(ensemble)) * dr)
        intergration.append(intergration[-1] + (yinterp[i] + yinterp[i - 1]) / 2. * (xvals[i] / get_a(ensemble)) * dr)
        # intergration.append(intergration[-1] + yinterp[i - 1] * (xvals[i] / get_a(ensemble)) * dr)
    intergration = 2. * np.pi ** 2. / 3. * (get_fpi(ensemble) / 2. ** (1. / 2.)) ** 2. * np.array(intergration)
    intergration = intergration[num_p * 3 / 4] - intergration
    return xvals, intergration


def get_fr_integration_mean_err(fr_all_config, ensemble, r_max, num_p=1000):
    length = len(fr_all_config[0])
    x = get_x_interpolation_axis(length, r_max)
    xvals = np.linspace(0, x[-1], num_p)
    fr_integration_list = []
    for fr in fr_all_config:
        _, fr_integration = get_fr_integration_one_traj(fr, ensemble, r_max, num_p)
        fr_integration_list.append(fr_integration)
    mean = np.mean(fr_integration_list, axis=0)
    err = 1. / len(fr_integration_list) ** (1. / 2.) * np.std(fr_integration_list, axis=0, ddof=1)
    return xvals, mean, err


def plt_x_fr_err(x, fr, err=None, label='', shape='*', color='r'):
    if err is None:
        plt.errorbar(x, fr, fmt=shape, label=label, color=color)
    else:
        plt.errorbar(x, fr, yerr=err, fmt=shape, label=label, color=color)
    return


if __name__ == '__main__':

    a_list = []
    fr_integ_list = []

    # model
    ensemble = 'physical-24nt96-1.0-interpolation'
    ensemble_ = 'physical-24nt96-1.0'
    fr = read_complex_bi(get_fr_model_path(ensemble))
    plt_fr_interpolation_mean_err(fr, 4., label=ensemble, shape='.', color='y')
    x, fr_integ = get_fr_integration_one_traj(fr, ensemble_, 4., num_p=200)
    plt_x_fr_err(x, fr_integ, label=ensemble, shape='.', color='y')

    # a_list.append(get_a(ensemble_))
    # fr_integ_list.append(fr_integ.real[:])

    ensemble = 'physical-32nt128-1.0-interpolation'
    ensemble_ = 'physical-32nt128-1.0'
    fr = read_complex_bi(get_fr_model_path(ensemble))
    plt_fr_interpolation_mean_err(fr, 4., label=ensemble, shape='.', color='g')
    x, fr_integ = get_fr_integration_one_traj(fr, ensemble_, 4., num_p=200)
    plt_x_fr_err(x, fr_integ, label=ensemble, shape='.', color='g')

    # a_list.append(get_a(ensemble_))
    # fr_integ_list.append(fr_integ.real[:])

    ensemble = 'physical-48nt192-1.0-interpolation'
    ensemble_ = 'physical-48nt192-1.0'
    fr = read_complex_bi(get_fr_model_path(ensemble))
    plt_fr_interpolation_mean_err(fr, 4., label=ensemble, shape='.', color='b')
    x, fr_integ = get_fr_integration_one_traj(fr, ensemble_, 4., num_p=200)
    plt_x_fr_err(x, fr_integ, label=ensemble, shape='.', color='b')

    a_list.append(get_a(ensemble_))
    fr_integ_list.append(fr_integ.real[:])

    ensemble = 'physical-32nt128-1.3333-interpolation'
    ensemble_ = 'physical-32nt128-1.3333'
    fr = read_complex_bi(get_fr_model_path(ensemble))
    plt_fr_interpolation_mean_err(fr, 4., label=ensemble, shape='.', color='r')
    x, fr_integ = get_fr_integration_one_traj(fr, ensemble_, 4., num_p=200)
    plt_x_fr_err(x, fr_integ, label=ensemble, shape='.', color='r')

    a_list.append(get_a(ensemble_))
    fr_integ_list.append(fr_integ.real[:])

    ensemble = 'physical-48nt192-2.0-interpolation'
    ensemble_ = 'physical-48nt192-2.0'
    fr = read_complex_bi(get_fr_model_path(ensemble))
    plt_fr_interpolation_mean_err(fr, 4., label=ensemble, shape='.', color='c')
    x, fr_integ = get_fr_integration_one_traj(fr, ensemble_, 4., num_p=200)
    plt_x_fr_err(x, fr_integ, label=ensemble, shape='.', color='c')

    a_list.append(get_a(ensemble_))
    fr_integ_list.append(fr_integ.real[:])

    fr_integ_list = np.array(fr_integ_list)
    fr_integ_extra, _ = get_a2_extrapolation_list(a_list, fr_integ_list)
    plt_x_fr_err(x, fr_integ_extra, label='fr_integ_extra', shape='.', color='black')

    x = [0, x[-1]]
    y = [1., 1.]
    plt.plot(x, y, '-')

    x = [0, x[-1]]
    y = [0., 0.]
    plt.plot(x, y, '-')

    plt.legend()
    plt.show()

    # lattice
    ensemble = '24D-0.00107-interpolation'
    ensemble_ = '24D-0.00107'
    fr = get_fr_all_traj(ensemble)
    fr_mean, fr_err = get_fr_mean_err(fr)
    plt_fr_interpolation_mean_err(fr_mean, 4., err=fr_err)
    x, y, err = get_fr_integration_mean_err(fr, ensemble_, 4., num_p=100)
    # plt_fr_interpolation_mean_err(y, 4., err=err)

    ensemble = '32D-0.00107-interpolation'
    ensemble_ = '32D-0.00107'
    fr = get_fr_all_traj(ensemble)
    x, y, err = get_fr_integration_mean_err(fr, ensemble_, 4., num_p=100)
    plt_x_fr_err(x, y, err, ensemble)

    ensemble = '32Dfine-0.0001-interpolation'
    ensemble_ = '32Dfine-0.0001'
    fr = get_fr_all_traj(ensemble)
    x, y, err = get_fr_integration_mean_err(fr, ensemble_, 4., num_p=100)
    plt_x_fr_err(x, y, err, ensemble)

    ensemble = '48I-0.00078-interpolation'
    ensemble_ = '48I-0.00078'
    fr = get_fr_all_traj(ensemble)
    x, y, err = get_fr_integration_mean_err(fr, ensemble_, 4., num_p=100)
    plt_x_fr_err(x, y, err, ensemble)

    x = [0, x[-1]]
    y = [1., 1.]
    plt.plot(x, y, '-')

    x = [0, x[-1]]
    y = [0., 0.]
    plt.plot(x, y, '-')

    plt.legend()
    plt.show()
    exit(0)

    '''
    # lattice
    ensemble = '24D-0.00107'
    fr = get_fr_all_traj(ensemble)
    mean, err = get_fr_mean_err(fr)
    plt_fr_mean_err(mean, get_a(ensemble), err=err, label=ensemble)

    ensemble = '32D-0.00107'
    fr = get_fr_all_traj(ensemble)
    mean, err = get_fr_mean_err(fr)
    plt_fr_mean_err(mean, get_a(ensemble), err=err, label=ensemble)

    ensemble = '32Dfine-0.0001'
    fr = get_fr_all_traj(ensemble)
    mean, err = get_fr_mean_err(fr)
    plt_fr_mean_err(mean, get_a(ensemble), err=err, label=ensemble)

    ensemble = '48I-0.00078'
    fr = get_fr_all_traj(ensemble)
    mean, err = get_fr_mean_err(fr)
    plt_fr_mean_err(mean, get_a(ensemble), err=err, label=ensemble)

    plt.legend()
    plt.show()
    exit(0)
    '''
