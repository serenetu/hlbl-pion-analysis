import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import jackknife as jk
import ensemble as es


def mod(x, length):
    m = x % length
    if 0 <= m:
        return m
    else:
        return m + len


def smod(x, length):
    m = mod(x, length)
    if m * 2 < length:
        return m
    else:
        return m - length


def smod_list(x_list, length):
    res = []
    for x in x_list:
        res.append(smod(x, length))
    return np.array(res)


def abs_smod_list(x_list, length):
    return np.absolute(smod_list(x_list, length))


def read_complex_bi(path):
    x = np.fromfile(path, dtype='complex128')
    return np.real(x)


class WallWallModelCorr:

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.corr_path = './data/WallWallCorr/' + self.ensemble + '/pion-corr.txt'
        return

    def read_corr(self):
        res = []
        f = open(self.corr_path)
        for i, line in enumerate(f):
            if i == 0:
                continue
            res.append(float(line.split()[1]))
        self.corr = np.array(res)
        return self.corr

    def get_zw(self):
        self.zw = (
                self.corr *
                np.exp(
                    abs_smod_list(range(len(self.corr)), len(self.corr)) *
                    es.get_mpi(self.ensemble)
                )
        )
        return

    def plt_zw(self, color='r', label=''):
        x = range(len(self.zw))
        plt.errorbar(x, self.zw, marker='x', color=color, label=label)


class WallWallCorr:

    def __init__(self, ensemble, accuracy):
        self.ensemble = ensemble
        self.accuracy = accuracy
        self.ensemble_accuracy = 'data/WallWallCorr/' + self.ensemble + '/' + self.accuracy

        if self.ensemble == '24D-0.00107':
            self.pion = 0.13975
            self.traj_start = 1000
            self.traj_end = 3000
        elif self.ensemble == '24D-0.0174':
            self.pion = 0.3357
            self.traj_start = 200
            self.traj_end = 1000
        elif self.ensemble == '32D-0.00107':
            self.pion = 0.139474
            self.traj_start = 680
            self.traj_end = 2000
        elif self.ensemble == '32Dfine-0.0001':
            self.pion = 0.10468
            self.traj_start = 200
            self.traj_end = 2000
        elif self.ensemble == '48I-0.00078':
            self.pion = 0.08049
            self.traj_start = 500
            self.traj_end = 3000

        self.__print_info()

    def __print_info(self):
        print('WallWallCorr:')
        print('Ensemble: ' + self.ensemble)
        print('Accuracy: ' + self.accuracy)
        print('Path: ' + self.ensemble_accuracy)
        print('Traj Start: ' + str(self.traj_start))
        print('Traj End: ' + str(self.traj_end))
        return

    def __read_traj(self, traj):
        path = self.ensemble_accuracy + '/results={0:04d}'.format(traj)
        if not os.path.isfile(path):
            return None
        return read_complex_bi(path)

    def read_all_traj(self):
        self.traj_to_corr = {}
        for traj in range(self.traj_end, self.traj_start - 1, -10):
            x = self.__read_traj(traj)
            if x is None: continue
            print('traj=' + str(traj) + ' have been read')
            self.traj_to_corr[traj] = x
        print(str(len(self.traj_to_corr)) + ' trajs have been read')

    def __compute_jk_dict(self):
        self.jk_traj_to_corr = jk.make_jackknife_dict(self.traj_to_corr)
        return

    def compute_corr_mean_err(self):
        self.__compute_jk_dict()
        self.corr_mean, self.corr_err = jk.jackknife_avg_err_dict(self.jk_traj_to_corr)
        print('wall wall corr:')
        print(self.corr_mean)
        print(self.corr_err)
        return

    def compute_zw_mean_err(self):
        self.zw_mean = self.corr_mean * np.exp(np.array(range(len(self.corr_mean))) * self.pion)
        self.zw_err = self.corr_err * np.exp(np.array(range(len(self.corr_err))) * self.pion)
        print('zw:')
        print(self.zw_mean)
        print(self.zw_err)

    def plt_zw(self, color, label):
        x = range(len(self.zw_mean))
        plt.errorbar(x, self.zw_mean, yerr=self.zw_err, marker='x', color=color, label=label)


if __name__ == '__main__':
    wwc_model = WallWallModelCorr('heavy-24nt96-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('heavy-32nt128-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('heavy-32nt128-1.3333')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('heavy-48nt192-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('heavy-48nt192-2.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)
    exit(0)

    wwc_model = WallWallModelCorr('physical-24nt96-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('physical-32nt128-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('physical-32nt128-1.3333')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('physical-48nt192-1.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)

    wwc_model = WallWallModelCorr('physical-48nt192-2.0')
    wwc_model.read_corr()
    wwc_model.get_zw()
    wwc_model.plt_zw()
    plt.show()
    print(wwc_model.zw)
    exit(0)


    wwc = WallWallCorr('48I-0.00078', 'ama')
    wwc.read_all_traj()
    wwc.compute_corr_mean_err()
    wwc.compute_zw_mean_err()
    wwc.plt_zw('r', '')
    plt.show()
    print(wwc.zw_mean[10])
    exit(0)

    wwc = WallWallCorr('24D-0.00107', 'ama')
    wwc.read_all_traj()
    wwc.compute_corr_mean_err()
    wwc.compute_zw_mean_err()
    wwc.plt_zw('r', '')
    plt.show()
    print(wwc.zw_mean[10])

    wwc = WallWallCorr('24D-0.0174', 'ama')
    wwc.read_all_traj()
    wwc.compute_corr_mean_err()
    wwc.compute_zw_mean_err()
    wwc.plt_zw('r', '')
    plt.show()
    print(wwc.zw_mean[10])

    wwc = WallWallCorr('32D-0.00107', 'ama')
    wwc.read_all_traj()
    wwc.compute_corr_mean_err()
    wwc.compute_zw_mean_err()
    wwc.plt_zw('r', '')
    plt.show()
    print(wwc.zw_mean[10])

    wwc = WallWallCorr('32Dfine-0.0001', 'ama')
    wwc.read_all_traj()
    wwc.compute_corr_mean_err()
    wwc.compute_zw_mean_err()
    wwc.plt_zw('r', '')
    plt.show()
    print(wwc.zw_mean[10])

    '''
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

    '''
