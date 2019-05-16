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

    def make_traj_pair_list(self):
        res = []
        for i in range(len(self.traj_list) / 2):
            ii = len(self.traj_list) / 2 + i
            res.append((self.traj_list[i], self.traj_list[ii]))
        return res

    def one_config_f2_valid(self, pair, num):
        if self.ama is True:
            pair_path = self.ensemble_path + "/traj=" + str(pair[0]).zfill(4) + "," + str(pair[1]).zfill(4) + ";ama;t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0"
        else:
            pair_path = self.ensemble_path + "/traj=" + str(pair[0]).zfill(4) + "," + str(pair[1]).zfill(4) + ";t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0;accuracy=0"
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
        if self.ama is True:
            pair_path = self.ensemble_path + "/traj=" + str(pair[0]).zfill(4) + "," + str(pair[1]).zfill(4) + ";ama;t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0"
        else:
            pair_path = self.ensemble_path + "/traj=" + str(pair[0]).zfill(4) + "," + str(pair[1]).zfill(4) + ";t-min=" + str(self.t_min).zfill(4) + ";xxp-limit=" + str(self.xxp_limit) + ";mod=" + str(self.mod) + ";type=0;accuracy=0"
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

    def __init__(self, ama=True, mod="", t_min=10, xxp_limit=10, m_pi=0.139474, zw=3.24457020e+08, zv=0.73464):
        self.ensemble = "32D-0.00107"
        self.ensemble_path = "/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/" + self.ensemble
        self.l = 32
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
        self.traj_list = range(680, 1371, 10)
        self.traj_list = range(680, 1371, 10)
        self.traj_pair_list = self.make_traj_pair_list()
        self.num_row = 80
        self.num_col = 40
        print "Traj Pair List"
        print self.traj_pair_list
        return


def do_lattice_analysis(ensemble="24D", mod="", t_min=10, xxp_limit=10, l=24, zw=1.28*10.**8., zv=0.73457):
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/" + ensemble + "/traj=1010,1660;t-min=0010;xxp-limit=10;mod=" + mod + ";type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(l=l, zw=zw, zv=zv)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')
    plt.show()


if __name__ == '__main__':
    # plot luchang table
    LUCHANG_PATH = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/test/tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 21)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0', label='pion pole model')

    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=16)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(15, 20))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 20), color='g', label='32D ama xxp_limit=16 jk (' + str(ana_32.num_configs) + ')')

    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=14)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(15, 20))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 20), color='b', label='32D ama xxp_limit=14 jk (' + str(ana_32.num_configs) + ')')

    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=10)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(15, 20))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 20), color='r', label='32D ama xxp_limit=10 jk (' + str(ana_32.num_configs) + ')')

    #plt.text(6, 4, 'preliminary', ha='left', wrap=True)
    plt.legend(loc='upper right')
    plt.xlabel('The longest distance between x, y, y\'')
    plt.ylabel('g-2 (partial sum from inf to 0)')
    plt.show()
    exit()

    ana_32 = Do32DLatticeAnalysis(mod='', ama=False)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(10,11))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 11), color='b', label='32D sloppy jk (' + str(ana_32.num_configs) + ')')

    ana_32 = Do32DLatticeAnalysis(mod='', ama=True)
    ana_32.get_all_config_f2(1024)
    # ana_32.plt_all_config_f2(color='b', rows=range(10,11))
    ana_32.get_f2_jk_dict()
    ana_32.plt_f2_jk(rows=range(10, 11), color='r', label='32D ama jk (' + str(ana_32.num_configs) + ')')

    exit()

    ana_32 = Do32DLatticeAnalysis(mod='', ama=False)
    ana_32.get_all_config_f2()
    ana_32.plt_all_config_f2(color='g', label='32D sloppy')

    ana_24 = Do24DLatticeAnalysis(mod='', ama=False)
    ana_24.get_all_config_f2()
    ana_24.plt_all_config_f2(color='r', label='24D sloppy')


    ana_32 = Do32DLatticeAnalysis(mod='', ama=False)
    ana_32.get_all_config_f2()
    ana_32.plt_all_config_f2(color='g', label='32D sloppy')

    ana_24 = Do24DLatticeAnalysis(mod='', ama=False)
    ana_24.get_all_config_f2()
    ana_24.plt_all_config_f2(color='r', label='24D sloppy')

    plt.legend(loc='upper right')

    ana_24.read_f2_jk_dict()
    ana_24.plt_f2_jk(color='b')



    ana_24 = Do24DLatticeAnalysis(mod='')
    ana_24.read_f2_jk_dict()
    ana_24.plt_f2_jk(color='r')

    ana_24 = Do24DLatticeAnalysis(mod='xyp>=xy')
    ana_24.read_f2_jk_dict()
    ana_24.plt_f2_jk(color='b')

    ana_24 = Do24DLatticeAnalysis(mod='xy>=xyp')
    ana_24.read_f2_jk_dict()
    ana_24.plt_f2_jk(color='g')

    plt.show()
    exit()

    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=10;mod=", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')

    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=10;mod=xyp>=xy", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')

    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=10;mod=xy>=xyp", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')
    plt.show()
    exit()


    PATH = '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/test/'
    ZV = 0.73457

    do_lattice_analysis(ensemble="24D", mod="", t_min=10, xxp_limit=10, l=24, zw=1.28*10.**8., zv=0.73457)
    exit()

    # plot luchang table
    LUCHANG_PATH = PATH + '/' + 'tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 30)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/24D/traj=1030,1030;t-min=0010;xxp-limit=10;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=10, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/24D/traj=1030,1030;t-min=0010;xxp-limit=10;mod=xy>=xyp;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=10, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/24D/traj=1030,1030;t-min=0010;xxp-limit=10;mod=xyp>=xy;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=10, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')
    plt.show()

    # plot luchang table
    LUCHANG_PATH = PATH + '/' + 'tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 30)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0')

    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=10;mod=", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')

    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=10;mod=xyp>=xy", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 15)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')
    plt.show()
    exit()






    # plot model tsep 2000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=2000;xxp-limit=25", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(10, 6)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='g')

    # plot model tsep 1000
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=1000;xxp-limit=10", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=2000)
    NUM_PAIRS = all_table.shape[0]
    UNIT = 0.2
    R_RANGE = range(5, 6)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    plt.show()
    exit()

    # plot model tsep 100
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=0100;xxp-limit=25", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=100)
    NUM_PAIRS = 360
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    # plot model tsep 40
    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/model/pion=0.139750;t-sep=0040;xxp-limit=25", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_model_factor(t=40)
    NUM_PAIRS = 182
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='b')

    # plot model tsep40
    MODEL_LABEL = 'pgge_model_tsep=40.'
    F_PATH_LABEL = PATH + '/' + MODEL_LABEL
    FACTOR = 10. ** 10. * set_model_factor(t=40)
    NUM_PAIRS = 296
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2(F_PATH_LABEL, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='g')
    plt.show()

    # plot luchang table
    LUCHANG_PATH = PATH + '/' + 'tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 30)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/32D-0.00107/traj=1050,1050;t-sep=0020;xxp-limit=20;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=20, l=32, zw=3.24457020e+08, zv=0.72)
    NUM_PAIRS = 900
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='b')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/32D-0.00107/traj=1050,1050;t-sep=0020;xxp-limit=15;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=20, l=32, zw=3.24457020e+08, zv=0.72)
    NUM_PAIRS = 900
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='g')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/24D/traj=1030,1030;t-sep=0020;xxp-limit=12;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=20, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = 700
    UNIT = 0.2
    R_RANGE = range(10, 20)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    all_table = read_all_bi_table("/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/data/f2/24D/traj=1030,1030;t-sep=0020;xxp-limit=10;type=0;accuracy=0", num_row=80, num_col=40)
    print all_table.shape
    FACTOR = 10. ** 10. * set_factor(t=20, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = 715
    UNIT = 0.2
    R_RANGE = range(10, 20)
    plot_f2_(all_table, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='black')
    plt.show()
    exit()

    # plot model
    MODEL_LABEL = 'pgge_model_rotate_from_y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096.'
    F_PATH_LABEL = PATH + '/' + MODEL_LABEL
    FACTOR = 10. ** 10. * set_model_factor(t=30)
    NUM_PAIRS = 374
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2(F_PATH_LABEL, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='r')

    # plot model tsep40
    MODEL_LABEL = 'pgge_model_tsep=40.'
    F_PATH_LABEL = PATH + '/' + MODEL_LABEL
    FACTOR = 10. ** 10. * set_model_factor(t=40)
    NUM_PAIRS = 296
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2(F_PATH_LABEL, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='g')

    # plot lattice
    LATTICE_LABEL = 'pgge_lattice_tsep=20.'
    F_PATH_LABEL = PATH + '/' + LATTICE_LABEL
    FACTOR = 10. ** 10. * set_factor(t=20, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    NUM_PAIRS = 856
    UNIT = 0.2
    R_RANGE = range(10, 20)
    plot_f2(F_PATH_LABEL, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='b')
    plt.show()
    exit(0)

    '''
    #FILE_LABEL = '512_rotation_model.'
    # plot old model
    MODEL_LABEL = '512_rotation_model.'
    F_PATH_LABEL = PATH + '/' + MODEL_LABEL
    FACTOR = 10 ** 10. * set_model_factor(t=30) * pion_prop(t=30) ** 2.
    NUM_PAIRS = 280
    UNIT = 0.2
    R_RANGE = range(10, 30)
    plot_f2(F_PATH_LABEL, NUM_PAIRS, UNIT, R_RANGE, factor=FACTOR, intg='infto0', color='y')

    #FILE_LABEL = 'distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:30.'
    #FILE_LABEL = 'y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512.'
    #FILE_LABEL = 'distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:1024,r_pion_to_gamma:30.'
    #FILE_LABEL = 'y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512_norotate.'
    #FILE_LABEL = 'pgge_rotate_from_y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096.'
    table_all = []
    for i in range(1, 66):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = one_table.transpose()
        one_table = partial_sum(one_table)
        # print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    factor = set_factor(t=20, l=24, zw=1.28*10.**8., zv=0.72) #* 0.0534983
    factor = set_model_factor(t=30)
    print('factor: ', factor)
    table_avg = 10**10 * factor * np.average(table_all, axis=0)
    table_std = 10**10 * factor * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    #plt_table(table_avg, table_std, ylim=(0, 15), xlim=(0, 15), unit=0.2, rows=range(10, 20))
    #plt_table(table_avg[:, ::-1], table_std[:, ::-1], unit=0.2, rows=range(1, 20))
    plt_table(table_avg, table_std, unit=0.2, rows=range(10, 20))
    plt.show()
    '''


    '''
    # partial sum from 0 max_R to inf
    PATH= '/Users/tucheng/Desktop/Physics/research/light-by-light/res/test/'
    FILE_LABEL = '512.'

    luchang_table = read_table_noimag(PATH + '/' + 'tab.txt')
    plt_table(luchang_table, unit=0.1, rows=range(20, 30))

    table_all = []
    for i in range(1, 290):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = partial_sum(one_table.transpose())
        print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    print('factor: ', set_factor(Z_pi = 1, Z_v = 1.))
    table_avg = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.average(table_all, axis=0)
    table_std = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    plt_table(table_avg, table_std, ylim=(0, 15), xlim=(0, 15), unit=0.2, rows=range(10, 20))
    plt.show()
    '''

    '''
    # partial sum from 0 max_R to inf
    PATH= '/Users/tucheng/Desktop/Physics/research/light-by-light/res/test/'
    FILE_LABEL = '512_model.'

    luchang_table = read_table_noimag(PATH + '/' + 'tab.txt')
    luchang_table = de_partial_sum(luchang_table)
    luchang_table = luchang_table[:, ::-1]
    luchang_table = partial_sum(luchang_table)
    plt_table(luchang_table[:, ::-1], unit=0.1, rows=range(20, 40))

    table_all = []
    for i in range(1, 290):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = partial_sum(one_table.transpose()[:, ::-1])
        print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    print('factor: ', set_factor(Z_pi = 1, Z_v = 1.))
    table_avg = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.average(table_all, axis=0)
    table_std = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    #plt_table(table_avg[:, ::-1], table_std[:, ::-1], ylim=(0, 15), xlim=(0,15), unit=0.2, rows=range(15, 20))
    plt_table(table_avg[:, ::-1], table_std[:, ::-1], unit=0.2, rows=range(15, 20))
    plt.show()
    '''


    '''
    # partial sum from inf max_R to 0
    PATH= '/Users/tucheng/Desktop/Physics/research/light-by-light/res/test/'
    FILE_LABEL = '1024_rerun_parsum.'

    luchang_table = read_table_noimag(PATH + '/' + 'tab.txt')
    luchang_table = de_partial_sum(luchang_table)
    luchang_table = luchang_table[:, ::-1]
    luchang_table = partial_sum(luchang_table)
    plt_table(luchang_table[:, ::-1], unit=0.1, rows=range(20, 40))

    table_all = []
    for i in range(1, 26):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = one_table.transpose()
        #one_table = de_partial_sum(one_table)
        #one_table = partial_sum(one_table[:, ::-1])
        #one_table = partial_sum(one_table)
        print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    print('factor: ', set_factor(Z_pi = 1, Z_v = 1.))
    table_avg = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.average(table_all, axis=0)
    table_std = 10**10 * set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.) * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    #plt_table(table_avg[:, ::-1], table_std[:, ::-1], ylim=(0, 10), xlim=(0,15), unit=0.2, rows=range(10, 21))
    plt_table(table_avg, table_std, unit=0.2, rows=range(10, 20))
    plt.show()
    '''

    '''
    # partial sum from inf max_R to 0
    PATH= '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/test/'
    FILE_LABEL = '512_model.'
    FILE_LABEL = '512_xbl_bmm.'
    FILE_LABEL = '512_xbm_bml.'
    #FILE_LABEL = '512_rotation_model.'

    luchang_table = read_table_noimag(PATH + '/' + 'tab.txt')
    plt_table(luchang_table, unit=0.1, rows=range(20, 40))

    table_all = []
    for i in range(1, 500):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = partial_sum(one_table.transpose())
        print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    factor = set_factor(mu_mass=0.1056583745/1.015, Z_pi = 1, Z_v = 1.)
    table_avg = 10**10 * factor * np.average(table_all, axis=0)
    table_std = 10**10 * factor * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    plt_table(table_avg, table_std, unit=0.2, rows=range(10, 20))
    plt.show()
    '''
