import sys
sys.path.append('/Users/tucheng/Desktop/Physics/research/qcdlib-python/qcdlib')
import numpy as np
import matplotlib.pyplot as plt

import jackknife as jk


def remove_0_label(line):
    line_fix = line.replace('[0] ', '')
    line_fix = line_fix.strip()
    return line_fix


def read_traj(line, num=4):
    line_fix = remove_0_label(line)
    _, traj = line_fix.split('traj=')
    return int(traj[:num])


def read_real(line):
    res = []
    line_fix = remove_0_label(line)
    vals = line_fix[:-1].split(',')
    for val in vals:
        real, imag = val.strip().split()
        res.append(float(real))
    return np.array(res)


def read_wall_to_wall_corr(fpath):
    res = {}
    f = open(fpath, 'r')
    if_read_real = False
    traj = None
    for line in f:
        line_fix = remove_0_label(line)
        if 'wall_wall_corr' in line_fix and 'result' in line_fix and 'traj' in line_fix:
            traj = read_traj(line_fix)
            if_read_real = True
            continue
        if if_read_real is True:
            assert traj is not None
            res[traj] = read_real(line)
            if_read_real = False
    return res


def read_zw(fpath):
    res = {}
    f = open(fpath, 'r')
    if_read_real = False
    traj = None
    for line in f:
        line_fix = remove_0_label(line)
        if 'zw' in line_fix and 'result' in line_fix and 'traj' in line_fix:
            traj = read_traj(line_fix)
            if_read_real = True
            continue
        if if_read_real is True:
            assert traj is not None
            res[traj] = read_real(line)
            if_read_real = False
    return res


def compute_zw_from_wall_wall_corr_dic(wall_wall_corr_dic, pion):
    res = {}
    for traj in wall_wall_corr_dic.keys():
        res[traj] = compute_zw_from_wall_wall_corr(wall_wall_corr_dic[traj], pion)
    return res



def compute_zw_from_wall_wall_corr(wall_wall_corr, pion):
    res = []
    for tsep in range(len(wall_wall_corr)):
        res.append(wall_wall_corr[tsep] * np.exp(tsep * pion))
    return np.array(res)



if __name__ == '__main__':
    WALL_TO_WALL_CORR_PATH = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/wall-to-wall-corr/2019.02.13-01:43:31-28860'
    #WALL_TO_WALL_CORR_PATH = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/wall-to-wall-corr/2019.01.27-10:47:08-102814'
    wall_wall_corr_dic = read_wall_to_wall_corr(WALL_TO_WALL_CORR_PATH)
    config_list = wall_wall_corr_dic.keys()
    config_list.sort()
    print 'Config List[' + str(len(wall_wall_corr_dic)) + ']:' + str(config_list)

    wall_wall_jk_dic = jk.make_jackknife_dic(wall_wall_corr_dic)
    wall_wall_mean, wall_wall_err = jk.jackknife_avg_err_dic(wall_wall_jk_dic)
    print wall_wall_mean
    print wall_wall_err
    plt.errorbar(range(len(wall_wall_mean)), wall_wall_mean, wall_wall_err, fmt='.', elinewidth=1, label='wall wall corr')
    plt.show()

    #PION = 0.13975 #24D
    PION = 0.139474 #32D
    zw_dic = compute_zw_from_wall_wall_corr_dic(wall_wall_corr_dic, PION)
    zw_jk_dic = jk.make_jackknife_dic(zw_dic)
    zw_mean, zw_err = jk.jackknife_avg_err_dic(zw_jk_dic)
    print zw_mean
    print zw_err
    plt.errorbar(range(len(zw_mean)), zw_mean, zw_err, fmt='.', elinewidth=1, label='zw')
    plt.show()
