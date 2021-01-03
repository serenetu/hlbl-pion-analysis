import numpy as np


def get_ainv(ensemble):  # in GeV
    if ensemble == '24D-0.00107':
        ainv = 1.015
    elif ensemble == '24D-0.0174':
        ainv = 1.015
    elif ensemble == '32D-0.00107':
        ainv = 1.015
    elif ensemble == '32Dfine-0.0001':
        ainv = 1.378
    elif ensemble == '48I-0.00078':
        ainv = 1.73
    else:
        return float(ensemble.split('-')[-1])
    return ainv


def get_a(ensemble):  # in fm
    ainv = get_ainv(ensemble)
    a = 1. / ainv * 0.197  # 1 GeV-1 = .197 fm
    return a


def get_l(ensemble):
    if ensemble == '24D-0.00107':
        l = 24
    elif ensemble == '32D-0.00107':
        l = 32
    elif ensemble == '32Dfine-0.0001':
        l = 32
    elif ensemble == '48I-0.00078':
        l = 48
    else:
        return int(ensemble.split('-')[1].split('nt')[0])
    return l


def get_mpi(ensemble):
    if ensemble == '24D-0.00107':
        m_pi = 0.13975
    elif ensemble == '32D-0.00107':
        m_pi = 0.139474
    elif ensemble == '32Dfine-0.0001':
        m_pi = 0.10468
    elif ensemble == '48I-0.00078':
        m_pi = 0.08049
    elif ensemble.split('-')[0] == 'physical':
        return np.arccosh(1. + (0.1349770 / get_ainv(ensemble)) ** 2. / 2.)
    elif ensemble.split('-')[0] == 'heavy':
        return np.arccosh(1. + (0.340 / get_ainv(ensemble)) ** 2. / 2.)
    return m_pi


def get_fpi(ensemble):
    if ensemble == '24D-0.00107':
        f_pi = 0.13055
    elif ensemble == '32D-0.00107':
        f_pi = 0.13122
    elif ensemble == '32Dfine-0.0001':
        f_pi = 0.09490
    elif ensemble == '48I-0.00078':
        f_pi = 0.07580
    elif 'physical-' in ensemble:
        return 0.092 * 2. ** (1. / 2.) / get_ainv(ensemble)
    elif 'heavy-' in ensemble:
        return 0.105 * 2. ** (1. / 2.) / get_ainv(ensemble)
    return f_pi


def get_zw(ensemble):
    if ensemble == '24D-0.00107':
        zw = 131683077.512
    elif ensemble == '32D-0.00107':
        zw = 319649623.111
    elif ensemble == '32Dfine-0.0001':
        zw = 772327306.431
    elif ensemble == '48I-0.00078':
        zw = 5082918150.729124
    elif ensemble == 'physical-24nt96-1.0':
        return 51092.89660689
    elif ensemble == 'physical-32nt128-1.0':
        return 121108.5192045
    elif ensemble == 'physical-32nt128-1.3333':
        return 161635.12552516
    elif ensemble == 'physical-48nt192-1.0':
        return 408741.22632696
    elif ensemble == 'physical-48nt192-2.0':
        return 818879.80926111
    elif ensemble == 'heavy-24nt96-1.0':
        return 20041.86944759
    elif ensemble == 'heavy-32nt128-1.0':
        return 47506.6535054
    elif ensemble == 'heavy-32nt128-1.3333':
        return 63733.40372925
    elif ensemble == 'heavy-48nt192-1.0':
        return 160334.95558074
    elif ensemble == 'heavy-48nt192-2.0':
        return 324101.87738791
    return zw


def get_zv(ensemble):
    if ensemble == '24D-0.00107':
        zv = 0.72672
    elif ensemble == '32D-0.00107':
        zv = 0.7260
    elif ensemble == '32Dfine-0.0001':
        zv = 0.68339
    elif ensemble == '48I-0.00078':
        zv = 0.71076
    return zv
