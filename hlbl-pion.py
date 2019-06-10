import sys
sys.path.append('/Users/tucheng/Desktop/Physics/research/qcdlib-python/qcdlib')
import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
import os
import jackknife as jk
import json


def read_table(fname):
    f = open(fname, 'r')
    res = []
    for line in f:
        if line == '\n':
            break
        res.append([])
        vals = line.strip().split()
        last_line = res[-1]
        for i in range(0, len(vals), 2):
            real, imag = vals[i], vals[i+1]
            #last_line.append(complex(float(real), float(imag)))
            last_line.append(float(real))
    return np.array(res)


def read_bi_table(f, num_row=0, num_col=0, val_in="complex", val_out="real"):
    fd = open(f, 'rb')
    if val_in == "complex":
        x = np.fromfile(fd, dtype=np.complex128)
    elif val_in == "real":
        x = np.fromfile(fd, dtype=np.float64)
    else:
        print "Unknown val_in Type"
        exit()
    x = x.reshape((num_row, num_col))
    if val_out == "real":
        x = x.real
    else:
        print "Unknown val_out Type"
        exit()
    return x


def read_all_bi_table(path, num_row=0, num_col=0, val_in="complex", val_out="real"):
    res = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        if not os.path.isfile(path + '/' + file):
            continue
        if file == '.DS_Store':
            continue
        res.append(read_bi_table(path + '/' + file, num_row, num_col, val_in, val_out))
    return np.array(res)


def read_table_noimag(fname):
    f = open(fname, 'r')
    res = []
    for line in f:
        if line == '\n':
            break
        res.append([])
        vals = line.strip().split()
        last_line = res[-1]
        for i in range(0, len(vals)):
            real = vals[i]
            last_line.append(float(real))
    return np.array(res)


def inverse_table(table_, inverse_row=False, inverse_col=False):
    table = table_
    if inverse_row is True:
        table = table[::-1]
    if inverse_col is True:
        n_row, n_col = table.shape
        for row in range(n_row):
            table[row] = table[row][::-1]
    return table


def partial_sum(table):
    assert isinstance(table, np.ndarray), "The Table Should Be In Type Numpy.ndarray"
    shape = table.shape
    assert len(shape) == 2, "The Table Should Only Be 2-D"
    res = np.zeros(shape, dtype=complex)
    n_row = shape[0]
    n_col = shape[1]
    for row in range(n_row):
        for col in range(n_col):
            if row == 0 and col == 0:
                res[0][0] = table[0][0]
            elif row == 0:
                res[0][col] = res[0][col - 1] + table[0][col]
            elif col == 0:
                res[row][0] = res[row - 1][0] + table[row][0]
            else:
                res[row][col] = res[row - 1][col] + res[row][col - 1] - res[row - 1][col - 1] + table[row][col]
    return res


def de_partial_sum(table):
    assert isinstance(table, np.ndarray), "The Table Should Be In Type Numpy.ndarray"
    shape = table.shape
    assert len(shape) == 2, "The Table Should Only Be 2-D"
    res = np.zeros(shape, dtype=complex)
    n_row = shape[0]
    n_col = shape[1]
    for row in range(n_row-1, -1, -1):
        for col in range(n_col-1, -1, -1):
            if row == 0 and col == 0:
                res[0][0] = table[0][0]
            elif row == 0:
                res[0][col] = table[0][col] - table[0][col - 1]
            elif col == 0:
                res[row][0] = table[row][0] - table[row - 1][0]
            else:
                res[row][col] = table[row][col] - table[row - 1][col] - table[row][col - 1] + table[row - 1][col - 1]
    return res


def plt_table(table, table_std=None, unit=1., rows=None, ylim=None, xlim=None, color=None, label=''):
    shape = table.shape
    n_row = shape[0]
    n_col = shape[1]
    x = np.arange(n_col) * unit
    if rows:
        row_list = rows
    else:
        row_list = range(n_row)
    for row in row_list:
        if table_std is None:
            plt.plot(x, table[row], marker='*', color=color, label=label)
        else:
            plt.errorbar(x, table[row], yerr=table_std[row], marker='*', color=color, label=label)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    return


def compute_zp(l, m_pi):
    return 1. / (2. * m_pi * l ** 3.)


def set_factor(l, m_pi, zw, zv, mu_mass=0.1056583745 / 1.015, e=0.30282212, three=3):
    q_u = 2./3.
    q_d = -1./3.
    zp = compute_zp(l, m_pi)
    fac = 2. * mu_mass * e ** 6. * three / 2. / 3. * (zv ** 4.) / (zp * zw) * 1. / 2. * (q_u ** 2. - q_d ** 2.) ** 2.
    return fac


def pion_prop(t, m):
    return m * kn(1, t * m) / (4. * np.pi ** 2. * t)


def set_model_factor(t, m_pi, mu_mass=0.1056583745 / 1.015, e=0.30282212, three=3):
    prop = pion_prop(t, m_pi)
    fac = 2. * mu_mass * e ** 6. * three / 2. / 3. / prop ** 2.
    return fac


def plot_f2(f_path_label, num_pairs, unit, r_range, factor=1.0, intg='0toinf', color=''):
    table_all = []
    for i in range(1, num_pairs+1):
        f_name = f_path_label + str(i).zfill(5)
        one_table = read_table(f_name)
        if intg == '0toinf':
            one_table = one_table.transpose()
        elif intg == 'infto0':
            one_table = one_table.transpose()[:, ::-1]
        one_table = partial_sum(one_table)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print('(num_pairs, r, R): ' + str(table_all.shape))
    table_avg = factor * np.average(table_all, axis=0)
    table_std = factor * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    if intg == '0toinf':
        plt_table(table_avg, table_std, unit=unit, rows=r_range, color=color)
    elif intg == 'infto0':
        plt_table(table_avg[:, ::-1], table_std[:, ::-1], unit=unit, rows=r_range, color=color)
    return


def plot_f2_(table_all_, num_pairs, unit, r_range, factor=1.0, intg='0toinf', color=''):
    table_all = []
    for i in range(0, num_pairs):
        one_table = table_all_[i][:]
        if intg == '0toinf':
            one_table = one_table.transpose()
        elif intg == 'infto0':
            one_table = one_table.transpose()[:, 50:0:-1]
        one_table = partial_sum(one_table)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print('(num_pairs, r, R): ' + str(table_all.shape))
    table_avg = np.average(factor * table_all, axis=0)
    table_std = np.std(factor * table_all, axis=0) * len(table_all) ** (-1./2.)
    if intg == '0toinf':
        plt_table(table_avg, table_std, unit=unit, rows=r_range, color=color)
    elif intg == 'infto0':
        plt_table(table_avg[:, ::-1], table_std[:, ::-1], unit=unit, rows=r_range, color=color)
    plt.plot(range(10), np.zeros(10))
    return


def plot_luchang_table(f_path, unit, r_range, intg='0toinf', label=''):
    luchang_table = read_table_noimag(f_path)
    if intg == '0toinf':
        plt_table(luchang_table, unit=unit, rows=r_range, label=label)
    elif intg == 'infto0':
        luchang_table = de_partial_sum(luchang_table)
        luchang_table = luchang_table[:, ::-1]
        luchang_table = partial_sum(luchang_table)
        plt_table(luchang_table[:, ::-1], unit=unit, rows=r_range, label=label)


class DoLatticeAnalysis(object):

    def __init__(self, ama=True, mod="", t_min=10, xxp_limit=10, zw=1.28*10.**8., zv=0.73457):
        self.ensemble
        self.ensemble_path
        self.l
        self.ama = ama
        self.mod = mod
        self.t_min = t_min
        self.xxp_limit = xxp_limit
        self.m_pi
        self.zw = zw
        self.zv = zv
        self.f2_jk_dict_path
        self.traj_list = []
        self.traj_pair_list = self.make_traj_pair_list()
        self.num_row = 80
        self.num_col = 40
        print "Traj Pair List"
        print self.traj_pair_list
        return

    def make_traj_pair_list(self, traj_start, traj_end, traj_jump, traj_sep):
        res = []
        traj_batch = traj_start
        while traj_batch < traj_end:
            for traj in range(traj_batch, traj_batch + traj_jump, traj_sep):
                if traj + traj_jump > traj_end:
                    break
                res.append((traj, traj + traj_jump))
            traj_batch += traj_jump * 2
        return res

    def get_traj_pair_folder_name(self, traj_pair):
        if self.ama is True:
            return "traj=" + str(traj_pair[0]).zfill(4) + "," + str(traj_pair[1]).zfill(4) + ";accuracy=ama;t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0"
        else:
            return "traj=" + str(traj_pair[0]).zfill(4) + "," + str(traj_pair[1]).zfill(4) + ";accuracy=sloppy;t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0"

    def one_config_f2_valid(self, pair, num):
        pair_path = self.ensemble_path + '/' + self.get_traj_pair_folder_name(pair)
        if not os.path.exists(pair_path):
            return False
        files = os.listdir(pair_path)
        count = 0
        for file in files:
            if not os.path.isfile(pair_path + '/' + file):
                continue
            if file == '.DS_Store':
                continue
            count += 1
        return count >= num

    def get_one_config_f2(self, pair, intg='infto0', par_start=0, par_end=50):
        print 'Get One Config f2:'
        pair_path = self.ensemble_path + '/' + self.get_traj_pair_folder_name(pair)
        print pair_path
        table_all_ = read_all_bi_table(pair_path, num_row=self.num_row, num_col=self.num_col)

        table_all = []
        num_pairs = table_all_.shape[0]
        for i in range(0, num_pairs):
            one_table = table_all_[i][:]
            if intg == '0toinf':
                one_table = one_table.transpose()
            elif intg == 'infto0':
                one_table = one_table.transpose()[:, par_start:par_end]
                one_table = one_table[:, ::-1]
            one_table = partial_sum(one_table)
            one_table = one_table[:, ::-1]
            table_all.append(one_table)
        table_all = np.array(table_all).real
        print('(num_pairs, r, R): ' + str(table_all.shape))
        factor = 10. ** 10. * set_factor(l=self.l, m_pi=self.m_pi, zw=self.zw, zv=self.zv)
        table_avg = np.average(factor * table_all, axis=0)
        table_std = np.std(factor * table_all, axis=0) * len(table_all) ** (-1./2.)
        # plt_table(table_avg, table_std, unit=0.2, rows=range(15), color='r')
        # plt.show()
        return table_avg, table_std

    def get_all_config_f2(self, num_in_each_config=1024):
        count = 0
        self.f2_config_dict = {}
        for pair in self.traj_pair_list:
            if not self.one_config_f2_valid(pair, num_in_each_config):
                print "config " + str(pair) + " are not valid"
                continue
            self.f2_config_dict[pair] = self.get_one_config_f2(pair)[0]
            count += 1
            if count is None:
                break
        self.num_configs = len(self.f2_config_dict)
        print 'Get All Config f2:'
        print 'num_configs: ' + str(self.num_configs)

    def plt_all_config_f2(self, unit=0.2, rows=range(10, 11), color='r', label=''):
        for config in self.f2_config_dict.keys():
            table_avg = self.f2_config_dict[config]
            plt_table(table_avg, unit=unit, rows=rows, color=color, label=label)
        return

    def get_f2_jk_dict(self):
        try:
            self.f2_config_dict
        except AttributeError:
            print 'Run get_all_config_f2 First'
            exit()
        self.f2_jk_dict = jk.make_jackknife_dic(self.f2_config_dict)
        self.f2_jk_mean, self.f2_jk_err = jk.jackknife_avg_err_dic(self.f2_jk_dict)
        return

    def save_f2_jk_dict(self, path=None):
        if path is None:
            save_path = self.f2_jk_dict_path
        else:
            save_path = path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for one_f2 in self.f2_jk_dict.keys():
            fname = save_path + '/' + str(one_f2[0]) + ',' + str(one_f2[1])
            np.save(fname, self.f2_jk_dict[one_f2])
            print 'Save to: ' + fname
        return

    def read_f2_jk_dict(self, path=None):
        if path is None:
            save_path = self.f2_jk_dict_path
        else:
            save_path = path
        self.f2_jk_dict = {}
        f_list = os.listdir(save_path + '/')
        for f in f_list:
            if os.path.isfile(save_path + '/' + f) and 'npy' in f:
                pairs = f.split('.')[0].split(',')
                key = (int(pairs[0]), int(pairs[1]))
                self.f2_jk_dict[key] = np.load(save_path + '/' + f)
        return

    def plt_f2_jk(self, unit=0.2, rows=range(10, 11), color='r', label=''):
        self.f2_jk_mean, self.f2_jk_err = jk.jackknife_avg_err_dic(self.f2_jk_dict)
        plt_table(self.f2_jk_mean, self.f2_jk_err, unit=unit, rows=rows, color=color, label=label)
        return


class Do24DLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama=False, mod="", t_min=10, xxp_limit=10, m_pi=0.13975, zw=1.28*10.**8., zv=0.73457):
        self.ensemble = "24D"
        self.ensemble_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/" + self.ensemble
        self.l = 24
        self.ama = ama
        self.mod = mod
        self.t_min = t_min
        self.xxp_limit = xxp_limit
        self.m_pi = m_pi
        self.zw = zw
        self.zv = zv
        if self.ama is True:
            self.f2_jk_dict_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/ana-data/f2/jk/" + self.ensemble + ";ama;t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0"
        else:
            self.f2_jk_dict_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/ana-data/f2/jk/" + self.ensemble + ";t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0;accuracy=0"
        self.traj_list = [
            1010, 1030, 1050, 1070, 1090, 1110, 1140, 1160, 1180, 1220, 1240, 1260, 1280, 1300, 1320, 1360, 1380, 1400,
            1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760,
            1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2120, 2140,
            2160, 2180, 2200, 2220, 2240, 2260, 2280]
        self.traj_list = [1800,1800]
        self.traj_pair_list = self.make_traj_pair_list()
        self.num_row = 80
        self.num_col = 40
        print "Traj Pair List"
        print self.traj_pair_list
        return


class Do32DLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        # basic info
        self.ensemble = "32D-0.00107"
        self.l = 32
        self.m_pi = 0.139474
        self.zw = 3.24457020e+08
        self.zv = 0.73464
        self.t_min = 10
        self.num_row = 80
        self.num_col = 40

        # input
        self.ama = ama
        self.mod = mod
        self.xxp_limit = xxp_limit

        # setup
        f2_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2"
        self.ensemble_path = f2_path + '/' + self.ensemble
        f2_jk_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/ana-data/f2/jk/"
        # self.f2_jk_dict_path = f2_jk_path + '/' + self.get_traj_pair_folder_name()
        self.traj_pair_list = self.make_traj_pair_list(680, 1370, 50, 10)
        print "Traj Pair List"
        print self.traj_pair_list
        return


if __name__ == '__main__':
    # plot luchang table
    LUCHANG_PATH = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/test/tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 21)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0', label='pion pole model')

    ana_24 = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=12)
    ana_24.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(15, 20))
    ana_24.get_f2_jk_dict()
    ana_24.plt_f2_jk(rows=range(10, 20), color='g', label='24D ama xxp_limit=10 jk (' + str(ana_24.num_configs) + ')')

    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=12)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(15, 20))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 20), color='g', label='32D ama xxp_limit=10 jk (' + str(ana_32.num_configs) + ')')

    plt.show()

    exit()
