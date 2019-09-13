import sys
import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
import os
import jackknife as jk


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


def set_factor(l, m_pi, zw, zv, mu_mass, e=0.30282212, three=3):
    q_u = 2./3.
    q_d = -1./3.
    zp = compute_zp(l, m_pi)
    fac = 2. * mu_mass * e ** 6. * three / 2. / 3. * (zv ** 4.) / (zp * zw) * 1. / 2. * (q_u ** 2. - q_d ** 2.) ** 2.
    return fac


def pion_prop(t, m):
    return m * kn(1, t * m) / (4. * np.pi ** 2. * t)


'''
def set_model_factor(t, m_pi, mu_mass=0.1056583745 / 1.015, e=0.30282212, three=3):
    prop = pion_prop(t, m_pi)
    fac = 2. * mu_mass * e ** 6. * three / 2. / 3. / prop ** 2.
    return fac
'''


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


def plot_luchang_table(f_path, unit, r_range, intg='0toinf', label='', color='r'):
    luchang_table = read_table_noimag(f_path)
    if intg == '0toinf':
        plt_table(luchang_table, unit=unit, rows=r_range, label=label, color=color)
    elif intg == 'infto0':
        luchang_table = de_partial_sum(luchang_table)
        luchang_table = luchang_table[:, ::-1]
        luchang_table = partial_sum(luchang_table)
        print('luchang table shape:')
        print(luchang_table.shape)
        plt_table(luchang_table[:, ::-1], unit=unit, rows=r_range, label=label, color=color)
    return


class DoLatticeAnalysis(object):

    def __init__(self, ama, mod, xxp_limit):
        # input
        self.ama = ama
        self.mod = mod
        self.xxp_limit = xxp_limit

        # constant parameters
        self.num_row = 80
        self.num_col = 40
        self.f2_path = './data/f2'

        # other parameters
        self.ensemble = None
        self.l = None
        self.ainv = None
        self.m_pi = None
        self.zw = None
        self.zv = None
        self.t_min = None

        '''
        self.traj_start = None
        self.traj_end = None
        self.traj_jump = None
        self.traj_sep = None
        '''

        # parameter compute
        self.ensemble_path = None
        self.muon = None
        self.a = None
        self.inf_cut = None
        self.traj_pair_list = None

        # never use
        # self.f2_jk_dict_path = None
        # self.traj_list = []
        # self.traj_pair_list = self.make_traj_pair_list()
        return

    def compute_parameters(self):
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.muon = 0.1056583745 / self.ainv
        self.a = 1. / self.ainv * 0.197  # 1 GeV-1 = .197 fm
        self.inf_cut = int(10. / self.a)  # 10 fm

        # self.traj_pair_list = self.read_traj_pair_list()

        '''
        self.traj_pair_list = self.make_traj_pair_list(
            self.traj_start, self.traj_end,
            self.traj_jump, self.traj_sep
        )
        '''
        return

    def read_traj_pair_list(self):
        res = []
        file_name = ''
        if self.ensemble == "24D-0.00107":
            file_name = 'ensemble:' + self.ensemble + '_start:1000_end:3000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == '24D-0.0174':
            file_name = 'ensemble:' + self.ensemble + '_start:200_end:1000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == '32D-0.00107':
            file_name = 'ensemble:' + self.ensemble + '_start:680_end:2000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == "32Dfine-0.0001":
            file_name = 'ensemble:' + self.ensemble + '_start:200_end:2000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == '48I-0.00078':
            file_name = 'ensemble:' + self.ensemble + '_start:500_end:3000_step:10_numpairs:10000_seplimit:50'
        else:
            raise BaseException('No Such Ensemble')

        file_path = './TrajPairs/' + file_name
        f = open(file_path, 'r')
        for line in f:
            line = line.strip('\n')
            one_pair = line.split(' ')
            res.append((int(one_pair[0]), int(one_pair[1])))
        return res

    def show_info(self):
        print(self.__class__.__name__ + '::show_info():')
        print('========================================')
        print('ENSEMBLE: ' + str(self.ensemble))
        print('L: ' + str(self.l))
        print('AINV(GeV): ' + str(self.ainv))
        print('Pion Mass(Lattice Unit): ' + str(self.m_pi))
        print('ZW: ' + str(self.zw))
        print('ZV: ' + str(self.zv))
        print('T_MIN(Lattice Unit): ' + str(self.t_min))
        print('========================================')
        print('AMA: ' + str(self.ama))
        print('MOD: ' + str(self.mod))
        print('XXP_LIMIT: ' + str(self.xxp_limit))
        print('========================================')
        print('TABLE NUM ROW: ' + str(self.num_row))
        print('TABLE NUM COL: ' + str(self.num_col))
        print('F2 PATH: ' + str(self.f2_path))
        print('========================================')
        '''
        print('TRAJ_START: ' + str(self.traj_start))
        print('TRAJ_END: ' + str(self.traj_end))
        print('TRAJ_JUMP: ' + str(self.traj_jump))
        print('TRAJ_SEP: ' + str(self.traj_sep))
        print('========================================')
        '''
        print('ENSEMBLE PATH: ' + str(self.ensemble_path))
        print('MUON MASS(Lattice Unit): ' + str(self.muon))
        print('LATTICE SPACING (fm): ' + str(self.a))
        print('INF CUT (Lattice Unit): ' + str(self.inf_cut))
        print('TRAJ PAIR LIST:')
        print(self.traj_pair_list)
        print('========================================')
        print('')
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

    def get_one_config_f2(self, pair, par_start, par_end, intg='infto0'):
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
        factor = 10. ** 10. * set_factor(l=self.l, m_pi=self.m_pi, zw=self.zw, zv=self.zv, mu_mass=self.muon)
        table_avg = np.average(factor * table_all, axis=0)
        table_std = np.std(factor * table_all, axis=0) * len(table_all) ** (-1./2.)
        # plt_table(table_avg, table_std, unit=0.2, rows=range(15), color='r')
        # plt.show()
        return table_avg, table_std

    def get_all_config_f2(self, num_in_each_config):
        count = 0
        self.f2_config_dict = {}
        for pair in self.traj_pair_list:
            if not self.one_config_f2_valid(pair, num_in_each_config):
                print "config " + str(pair) + " are not valid"
                continue
            self.f2_config_dict[pair] = self.get_one_config_f2(
                pair, par_start=0, par_end=self.inf_cut)[0]
            count += 1
            if count is None:
                break
        self.num_configs = len(self.f2_config_dict)
        print 'Get All Config f2:'
        print 'num_configs: ' + str(self.num_configs)
        return

    def load_one_config_f2(self, pair, f2_path):
        ensemble_f2_path = f2_path + '/' + self.ensemble
        file_path = ensemble_f2_path + '/' + self.get_traj_pair_folder_name(pair)
        if not os.path.exists(file_path):
            return
        self.f2_config_dict[pair] = np.loadtxt(file_path)
        print('Load f2 from: ' + file_path)
        return

    def load_all_config_f2(self, f2_path):
        self.f2_config_dict = {}
        for pair in self.traj_pair_list:
            self.load_one_config_f2(pair, f2_path)
        print('Load All Config f2: {} pairs'.format(len(self.f2_config_dict)))
        return

    def save_one_config_f2(self, pair, f2_path):
        ensemble_f2_path = f2_path + '/' + self.ensemble
        if not os.path.exists(ensemble_f2_path):
            os.mkdir(ensemble_f2_path)
        file_path = ensemble_f2_path + '/' + self.get_traj_pair_folder_name(pair)
        np.savetxt(file_path, self.f2_config_dict[pair])
        print('Save One Pair f2 to: ' + file_path)
        return

    def save_all_config_f2(self, f2_path):
        for pair in self.traj_pair_list:
            if pair not in self.f2_config_dict:
                continue
            self.save_one_config_f2(pair, f2_path)
        return

    def get_f2_mean_err(self):
        array = np.array(self.f2_config_dict.values())
        self.f2_mean = np.mean(array, axis=0)
        print('length f2 config dict')
        print(len(self.f2_config_dict))
        self.f2_err = 1. / len(self.f2_config_dict) ** (1. / 2.) * np.std(array, axis=0)
        return

    def plt_all_config_f2(self, rows=range(10, 11), color='r', label=''):
        for config in self.f2_config_dict.keys():
            table_avg = self.f2_config_dict[config]
            plt_table(table_avg, unit=self.a, rows=rows, color=color, label=label)
        return

    def get_f2_jk_dict(self):
        '''
        '''
        '''
        try:
            self.f2_config_dict
        except AttributeError:
            print 'Run get_all_config_f2 First'
            exit()
        self.f2_jk_dict = jk.make_jackknife_dict(self.f2_config_dict)
        '''
        config_set = set()
        for pair in self.f2_config_dict:
            config_set.add(pair[0])
            config_set.add(pair[1])

        for config in config_set:
            array = []
            for pair in self.f2_config_dict:
                if config in pair:
                    continue
                array.append(self.f2_config_dict[pair])
            array = np.array(array)
            mean = np.mean(array, axis=0)
            self.f2_jk_dict[config] = mean
        return

    def get_f2_jk_mean_err(self):
        self.f2_jk_mean, self.f2_jk_err = jk.jackknife_avg_err_dict(self.f2_jk_dict)
        return

    def save_f2_jk_mean_err(self, path):
        print('f2 jk mean err:')
        print(self.f2_jk_mean)
        print(self.f2_jk_err)
        np.savetxt(path + '/' + self.jk_mean_err_save_label + '.mean', self.f2_jk_mean)
        np.savetxt(path + '/' + self.jk_mean_err_save_label + '.err', self.f2_jk_err)
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

    def plt_f2_jk(self, rows=range(10, 11), color='r'):
        # self.f2_jk_mean, self.f2_jk_err = jk.jackknife_avg_err_dict(self.f2_jk_dict)
        plt_table(self.f2_jk_mean, self.f2_jk_err, unit=self.a, rows=rows, color=color, label=self.jk_mean_err_save_label)
        return

    def plt_f2(self, rows=range(10, 11), color='r'):
        label = self.get_mean_err_save_label()
        plt_table(self.f2_mean, self.f2_err, unit=self.a, rows=rows, color=color, label=label)
        return

    def get_mean_err_save_label(self):
        if self.ama:
            ama_label = 'ama'
        else:
            ama_label = 'sloppy'
        res = '{};{};xxp_limit={};mod={};({})'.format(
            self.ensemble,
            ama_label,
            self.xxp_limit,
            self.mod,
            self.num_configs
        )
        mean_err_save_label = res
        return mean_err_save_label

    def save_f2_mean_err(self, path):
        print('f2 mean err:')
        print(self.f2_mean)
        print(self.f2_err)
        label = self.get_mean_err_save_label()
        np.savetxt(path + '/' + label + '.mean', self.f2_mean)
        np.savetxt(path + '/' + label + '.err', self.f2_err)
        return

    def get_jk_mean_err_save_label(self):
        if self.ama:
            ama_label = 'ama'
        else:
            ama_label = 'sloppy'
        res = '{};{};xxp_limit={};mod={};jk({})'.format(
            self.ensemble,
            ama_label,
            self.xxp_limit,
            self.mod,
            self.num_configs
        )
        self.jk_mean_err_save_label = res
        return self.jk_mean_err_save_label

    def read_f2_jk_mean_err(self, mean_path, err_path, label):
        self.jk_mean_err_save_label = label
        self.f2_jk_mean = np.loadtxt(mean_path)
        self.f2_jk_err = np.loadtxt(err_path)
        return


class Do24DLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.00107"
        self.l = 24
        self.ainv = 1.015
        self.m_pi = 0.13975  # lattice unit
        self.zw = 131683077.512
        self.zv = 0.72672
        self.t_min = 10

        self.traj_start = 1000
        self.traj_end = 2640
        self.traj_jump = 50
        self.traj_sep = 10

        self.compute_parameters()

        self.show_info()
        return


class Do24DHeavyLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DHeavyLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.0174"
        self.l = 24
        self.ainv = 1.015
        self.m_pi = 0.3357  # lattice unit
        self.zw = 58760419.01434206
        self.zv = 0.72672
        self.t_min = 10

        self.traj_start = 200
        self.traj_end = 560
        self.traj_jump = 50
        self.traj_sep = 10

        self.compute_parameters()

        self.show_info()
        return


class Do32DLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32D-0.00107"
        self.l = 32
        self.ainv = 1.015
        self.m_pi = 0.139474  # lattice unit
        self.zw = 319649623.111
        self.zv = 0.7260
        self.t_min = 10

        self.traj_start = 680
        self.traj_end = 1370
        self.traj_jump = 50
        self.traj_sep = 10

        self.compute_parameters()

        self.show_info()
        return


class Do32DfineLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DfineLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32Dfine-0.0001"
        self.l = 32
        self.ainv = 1.378
        self.m_pi = 0.10468
        self.zw = 772327306.431
        self.zv = 0.68339
        self.t_min = 14

        '''
        self.traj_start = 100
        self.traj_end = 430
        self.traj_jump = 50
        self.traj_sep = 10
        '''

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()

        self.show_info()
        return


if __name__ == '__main__':
    # plot luchang table
    LUCHANG_PATH = './data/f2/luchang/tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 21)
    plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0', label='pion pole model', color='black')

    path_save_f2_jk = './ana-data/f2/jk'
    path_save_f2 = './ana-data/f2/mean_err'
    path_save_all_config_f2 = './ana-data/f2'

    xxp_limit = 16
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    # ana_32_fine.get_all_config_f2(1024)
    # ana_32_fine.save_all_config_f2(path_save_all_config_f2)
    ana_32_fine.load_all_config_f2(path_save_all_config_f2)
    exit(0)

    xxp_limit = 10
    ana_24d_h = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d_h.get_all_config_f2(1024)
    ana_24d_h.get_f2_mean_err()
    ana_24d_h.save_f2_mean_err(path_save_f2)
    ana_24d_h.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_24d_h = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d_h.get_all_config_f2(1024)
    ana_24d_h.get_f2_mean_err()
    ana_24d_h.save_f2_mean_err(path_save_f2)
    ana_24d_h.plt_f2(rows=range(15, 20), color='b')
    plt.show()
    exit()

    xxp_limit = 10
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.get_all_config_f2(1024)
    ana_32d.get_f2_mean_err()
    ana_32d.save_f2_mean_err(path_save_f2)
    ana_32d.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.get_all_config_f2(1024)
    ana_32d.get_f2_mean_err()
    ana_32d.save_f2_mean_err(path_save_f2)
    ana_32d.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 14
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.get_all_config_f2(1024)
    ana_32d.get_f2_mean_err()
    ana_32d.save_f2_mean_err(path_save_f2)
    ana_32d.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 16
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.get_all_config_f2(1024)
    ana_32d.get_f2_mean_err()
    ana_32d.save_f2_mean_err(path_save_f2)
    ana_32d.plt_f2(rows=range(15, 20), color='b')
    plt.show()
    exit()

    xxp_limit = 10
    ana_24d = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d.get_all_config_f2(1024)
    ana_24d.get_f2_mean_err()
    ana_24d.save_f2_mean_err(path_save_f2)
    ana_24d.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_24d = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d.get_all_config_f2(1024)
    ana_24d.get_f2_mean_err()
    ana_24d.save_f2_mean_err(path_save_f2)
    ana_24d.plt_f2(rows=range(15, 20), color='b')
    plt.show()
    exit()

    xxp_limit = 10
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_mean_err()
    ana_32_fine.save_f2_mean_err(path_save_f2)
    ana_32_fine.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_mean_err()
    ana_32_fine.save_f2_mean_err(path_save_f2)
    ana_32_fine.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 14
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_mean_err()
    ana_32_fine.save_f2_mean_err(path_save_f2)
    ana_32_fine.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 16
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_mean_err()
    ana_32_fine.save_f2_mean_err(path_save_f2)
    ana_32_fine.plt_f2(rows=range(15, 20), color='b')

    '''
    xxp_limit = 10
    fname = '24D-0.00107;ama;xxp_limit=10;mod=;jk(11)'
    ana_24d = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                path_save_f2_jk + '/' + fname + '.err',
                                fname)
    ana_24d.plt_f2_jk(rows=range(15, 20), color='b')

    xxp_limit = 12
    fname = '24D-0.00107;ama;xxp_limit=12;mod=;jk(11)'
    ana_24d = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                path_save_f2_jk + '/' + fname + '.err',
                                fname)
    ana_24d.plt_f2_jk(rows=range(12, 13), color='g')

    xxp_limit = 10
    fname = '32D-0.00107;ama;xxp_limit=10;mod=;jk(20)'
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                path_save_f2_jk + '/' + fname + '.err',
                                fname)
    ana_32d.plt_f2_jk(rows=range(15, 20), color='g')

    xxp_limit = 12
    fname = '32D-0.00107;ama;xxp_limit=12;mod=;jk(20)'
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                path_save_f2_jk + '/' + fname + '.err',
                                fname)
    ana_32d.plt_f2_jk(rows=range(15, 20), color='b')

    xxp_limit = 14
    fname = '32D-0.00107;ama;xxp_limit=14;mod=;jk(20)'
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                path_save_f2_jk + '/' + fname + '.err',
                                fname)
    ana_32d.plt_f2_jk(rows=range(15, 20), color='r')
    '''

    '''
    xxp_limit = 10
    fname = '32Dfine-0.0001;ama;xxp_limit=10;mod=;jk(5)'
    ana_32dfine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32dfine.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                    path_save_f2_jk + '/' + fname + '.err',
                                    fname)
    ana_32dfine.plt_f2_jk(rows=range(15, 20), color='r')

    xxp_limit = 12
    fname = '32Dfine-0.0001;ama;xxp_limit=12;mod=;jk(5)'
    ana_32dfine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32dfine.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                    path_save_f2_jk + '/' + fname + '.err',
                                    fname)
    ana_32dfine.plt_f2_jk(rows=range(15, 20), color='g')

    xxp_limit = 14
    fname = '32Dfine-0.0001;ama;xxp_limit=14;mod=;jk(5)'
    ana_32dfine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32dfine.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                    path_save_f2_jk + '/' + fname + '.err',
                                    fname)
    ana_32dfine.plt_f2_jk(rows=range(15, 20), color='b')

    xxp_limit = 16
    fname = '32Dfine-0.0001;ama;xxp_limit=16;mod=;jk(5)'
    ana_32dfine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32dfine.read_f2_jk_mean_err(path_save_f2_jk + '/' + fname + '.mean',
                                    path_save_f2_jk + '/' + fname + '.err',
                                    fname)
    ana_32dfine.plt_f2_jk(rows=range(16, 17), color='y')
    '''

    '''
    # compute and save jk
    xxp_limit = 10
    ana_24 = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24.get_all_config_f2(1024)
    ana_24.get_f2_jk_dict()
    ana_24.get_f2_jk_mean_err()
    ana_24.get_jk_mean_err_save_label()
    ana_24.save_f2_jk_mean_err(path_save_f2_jk)
    ana_24.plt_f2_jk(rows=range(15, 20), color='g')

    xxp_limit = 12
    ana_24 = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24.get_all_config_f2(1024)
    # ana_24.get_f2_jk_dict()
    # ana_24.get_f2_jk_mean_err()
    # ana_24.get_jk_mean_err_save_label()
    # ana_24.save_f2_jk_mean_err(path_save_f2_jk)
    # ana_24.plt_f2_jk(rows=range(10, 20), color='g')
    ana_24.get_f2_mean_err()
    ana_24.save_f2_mean_err(path_save_f2)
    ana_24.plt_f2(rows=range(15, 20), color='g')

    xxp_limit = 10
    ana_24_h = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_h.get_all_config_f2(1024)
    ana_24_h.get_f2_mean_err()
    ana_24_h.save_f2_mean_err(path_save_f2)
    ana_24_h.plt_f2(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_24_h = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_h.get_all_config_f2(1024)
    ana_24_h.get_f2_mean_err()
    ana_24_h.save_f2_mean_err(path_save_f2)
    ana_24_h.plt_f2(rows=range(15, 20), color='b')
    '''

    '''
    xxp_limit = 10
    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32.get_all_config_f2(1024)
    ana_32.get_f2_jk_dict()
    ana_32.get_f2_jk_mean_err()
    ana_32.get_label()
    ana_32.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32.plt_f2_jk(rows=range(10, 20), color='r')

    xxp_limit = 12
    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32.get_all_config_f2(1024)
    ana_32.get_f2_jk_dict()
    ana_32.get_f2_jk_mean_err()
    ana_32.get_label()
    ana_32.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32.plt_f2_jk(rows=range(10, 20), color='r')

    xxp_limit = 14
    ana_32 = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32.get_all_config_f2(1024)
    ana_32.get_f2_jk_dict()
    ana_32.get_f2_jk_mean_err()
    ana_32.get_label()
    ana_32.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32.plt_f2_jk(rows=range(10, 20), color='r')
    '''

    '''
    xxp_limit = 10
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.get_jk_mean_err_save_label()
    ana_32_fine.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32_fine.plt_f2_jk(rows=range(15, 20), color='b')

    xxp_limit = 12
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.get_jk_mean_err_save_label()
    ana_32_fine.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32_fine.plt_f2_jk(rows=range(15, 20), color='b')

    xxp_limit = 14
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.get_jk_mean_err_save_label()
    ana_32_fine.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32_fine.plt_f2_jk(rows=range(15, 20), color='b')

    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.get_label()
    ana_32_fine.save_f2_jk_mean_err(path_save_f2_jk)
    ana_32_fine.plt_f2_jk(rows=range(10, 20), color='b')
    '''

    plt.show()
    exit()
