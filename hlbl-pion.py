import sys
import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
import os
import jackknife as jk
import ensemble as es
import extrapolation as ex


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


def save_with_err(path, x, y, y_err):
    f = open(path, 'w')
    for i in range(len(x)):
        f.write("{:.10E} {:.10E} {:.10E}\n".format(x[i], y[i], y_err[i]))
    f.close()
    return


def save_without_err(path, x, y):
    f = open(path, 'w')
    for i in range(len(x)):
        f.write("{:.10E} {:.10E}\n".format(x[i], y[i]))
    f.close()
    return


def save_table_without_err(path, table, unit, row):
    n_col = len(table[row])
    x = np.arange(n_col) * unit
    y = table[row]
    save_without_err(path, x, y)
    return


def save_table_with_err(path, table, table_err, unit, row):
    n_col = len(table[row])
    x = np.arange(n_col) * unit
    y = table[row]
    y_err = table_err[row]
    save_with_err(path, x, y, y_err)
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
        return luchang_table
    elif intg == 'infto0':
        luchang_table = de_partial_sum(luchang_table)
        luchang_table = luchang_table[:, ::-1]
        luchang_table = partial_sum(luchang_table)
        print('luchang table shape:')
        print(luchang_table.shape)
        plt_table(luchang_table[:, ::-1], unit=unit, rows=r_range, label=label, color=color)
        return luchang_table[:, ::-1]
    return


class DoLatticeAnalysis(object):

    def __init__(self, ama, mod, xxp_limit):
        # input
        self.ama = ama
        self.mod = mod
        self.xxp_limit = xxp_limit

        # constant parameters
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
        elif self.ensemble == "24D-0.00107-physical-pion":
            file_name = 'ensemble:24D-0.00107_start:1000_end:3000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == "24D-0.0174-physical-pion":
            file_name = 'ensemble:24D-0.0174_start:200_end:1000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == "32Dfine-0.0001-physical-pion":
            file_name = 'ensemble:32Dfine-0.0001_start:200_end:2000_step:10_numpairs:10000_seplimit:50'
        elif self.ensemble == '48I-0.00078-physical-pion':
            file_name = 'ensemble:48I-0.00078_start:500_end:3000_step:10_numpairs:10000_seplimit:50'
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
                one_table = one_table.transpose()[:, par_start:par_end + 1]
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

    def get_all_config_f2(self, num_in_each_config, num_pairs=None):
        self.f2_config_dict = {}
        for pair in self.traj_pair_list:
            if not self.one_config_f2_valid(pair, num_in_each_config):
                print "config " + str(pair) + " are not valid"
                continue
            if pair in self.f2_config_dict:
                continue
            self.f2_config_dict[pair] = self.get_one_config_f2(
                pair, par_start=0, par_end=self.inf_cut)[0]
            if len(self.f2_config_dict) == num_pairs:
                break
        print 'Get All Config f2:'
        print 'num_configs: ' + str(len(self.f2_config_dict))
        return

    def load_one_config_f2(self, pair, f2_path):
        ensemble_f2_path = f2_path + '/' + self.ensemble
        file_path = ensemble_f2_path + '/' + self.get_traj_pair_folder_name(pair)
        if not os.path.exists(file_path):
            return
        self.f2_config_dict[pair] = np.loadtxt(file_path)
        print('Load f2 from: ' + file_path)
        return

    def load_all_config_f2(self, f2_path, num_limit=None):
        self.f2_config_dict = {}
        for pair in self.traj_pair_list:
            self.load_one_config_f2(pair, f2_path)
            if len(self.f2_config_dict) == num_limit:
                break
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
        print(len(config_set), config_set)

        self.f2_jk_dict = {}
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

    def save_f2_jk_mean_err(self, row, path):
        save_table_with_err(path, self.f2_jk_mean, self.f2_jk_err, self.a, row)
        '''
        print('f2 jk mean err:')
        print(self.f2_jk_mean)
        print(self.f2_jk_err)
        np.savetxt(path + '/' + self.jk_mean_err_save_label + '.mean', self.f2_jk_mean)
        np.savetxt(path + '/' + self.jk_mean_err_save_label + '.err', self.f2_jk_err)
        '''
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

    def plt_f2_jk(self, rows=range(10, 11), color='r', label=''):
        # self.f2_jk_mean, self.f2_jk_err = jk.jackknife_avg_err_dict(self.f2_jk_dict)
        plt_table(self.f2_jk_mean, self.f2_jk_err, unit=self.a, rows=rows, color=color, label=label)
        return

    def plt_f2(self, rows=range(10, 11), color='r', label=''):
        # label = self.get_mean_err_save_label()
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
            len(self.f2_config_dict)
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
            len(self.f2_jk_dict)
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
        self.l = es.get_l(self.ensemble)
        self.ainv = es.get_ainv(self.ensemble)
        self.m_pi = es.get_mpi(self.ensemble)  # lattice unit
        self.zw = es.get_zw(self.ensemble)
        self.zv = es.get_zv(self.ensemble)
        self.t_min = 10

        self.num_row = 80
        self.num_col = 40

        # self.traj_start = 1000
        # self.traj_end = 2640
        # self.traj_jump = 50
        # self.traj_sep = 10

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()
        # self.traj_pair_list = self.make_traj_pair_list(1000, 2640, 50, 10)

        self.show_info()
        return


class Do24DPhysicalPionLatticeAnalysis(Do24DLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DPhysicalPionLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.00107-physical-pion"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class Do24DRefineFieldLatticeAnalysis(Do24DLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DRefineFieldLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.00107-refine-field"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class Do24DHeavyLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DHeavyLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.0174"
        self.l = es.get_l(self.ensemble)
        self.ainv = es.get_ainv(self.ensemble)
        self.m_pi = es.get_mpi(self.ensemble)  # lattice unit
        self.zw = es.get_zw(self.ensemble)
        self.zv = es.get_zv(self.ensemble)
        self.t_min = 10

        self.num_row = 80
        self.num_col = 40

        self.traj_start = 200
        self.traj_end = 560
        self.traj_jump = 50
        self.traj_sep = 10

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()

        self.show_info()
        return


class Do24DHeavyPhysicalPionLatticeAnalysis(Do24DHeavyLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do24DHeavyPhysicalPionLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "24D-0.0174-physical-pion"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class Do32DLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32D-0.00107"
        self.l = es.get_l(self.ensemble)
        self.ainv = es.get_ainv(self.ensemble)
        self.m_pi = es.get_mpi(self.ensemble)  # lattice unit
        self.zw = es.get_zw(self.ensemble)
        self.zv = es.get_zv(self.ensemble)
        self.t_min = 10

        self.num_row = 80
        self.num_col = 40

        self.traj_start = 680
        self.traj_end = 1370
        self.traj_jump = 50
        self.traj_sep = 10

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()

        self.show_info()
        return


class Do32DPhysicalPionLatticeAnalysis(Do32DLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DPhysicalPionLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32D-0.00107-physical-pion"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class Do32DfineLatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DfineLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32Dfine-0.0001"
        self.l = es.get_l(self.ensemble)
        self.ainv = es.get_ainv(self.ensemble)
        self.m_pi = es.get_mpi(self.ensemble)  # lattice unit
        self.zw = es.get_zw(self.ensemble)
        self.zv = es.get_zv(self.ensemble)
        self.t_min = 14

        '''
        self.traj_start = 100
        self.traj_end = 430
        self.traj_jump = 50
        self.traj_sep = 10
        '''

        self.num_row = 100
        self.num_col = 40

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()

        self.show_info()
        return


class Do32DfinePhysicalPionLatticeAnalysis(Do32DfineLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do32DfinePhysicalPionLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "32Dfine-0.0001-physical-pion"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class Do48ILatticeAnalysis(DoLatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do48ILatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "48I-0.00078"
        self.l = 48
        self.ainv = 1.73
        self.m_pi = 0.08049
        self.zw = 5082918150.729124
        self.zv = 0.71076
        self.t_min = 16

        self.num_row = 120
        self.num_col = 60

        self.compute_parameters()
        self.traj_pair_list = self.read_traj_pair_list()
        # self.traj_pair_list = self.make_traj_pair_list(1590, 3000, 60, 20)

        self.show_info()
        return


class Do48IPhysicalPionLatticeAnalysis(Do48ILatticeAnalysis):

    def __init__(self, ama, mod, xxp_limit):
        super(Do48IPhysicalPionLatticeAnalysis, self).__init__(ama, mod, xxp_limit)
        self.ensemble = "48I-0.00078-physical-pion"
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.show_info()
        return


class DoModelAnalysis(object):

    def __init__(self, ensemble):
        self.f2_path = './data/f2'

        self.ensemble = ensemble
        self.ensemble_path = self.f2_path + '/' + self.ensemble
        self.inf_cut = int(10. / es.get_a(self.ensemble))  # 10 fm

        self.num_row = 120
        self.num_col = 60
        return

    def get_a(self):
        return es.get_a(self.ensemble)

    def get_model_factor(self):
        e = 0.30282212
        three = 3
        muon = 0.1056583745 / es.get_ainv(self.ensemble)
        zp = compute_zp(es.get_l(self.ensemble), es.get_mpi(self.ensemble))
        zw = es.get_zw(self.ensemble)
        return 2. * muon * e ** 6. * three / 2. / 3. / (zp * zw)

    def get_model_f2(self, intg='infto0'):
        print 'Get Model f2:'
        pair_path = self.ensemble_path
        table_all_ = read_all_bi_table(pair_path, num_row=self.num_row, num_col=self.num_col)

        table_all = []
        num_pairs = table_all_.shape[0]
        for i in range(0, num_pairs):
            one_table = table_all_[i][:]
            if intg == '0toinf':
                one_table = one_table.transpose()
            elif intg == 'infto0':
                one_table = one_table.transpose()[:, 0:self.inf_cut + 1]
                one_table = one_table[:, ::-1]
            one_table = partial_sum(one_table)
            one_table = one_table[:, ::-1]
            table_all.append(one_table)
        table_all = np.array(table_all).real
        print('(num_pairs, r, R): ' + str(table_all.shape))
        table_avg = np.average(table_all, axis=0)
        table_std = np.std(table_all, axis=0) * len(table_all) ** (-1./2.)

        self.model_f2 = 10. ** 10. * table_avg * self.get_model_factor()
        self.model_f2_err = 10. ** 10. * table_std * self.get_model_factor()
        return

    def save_model_f2_table(self, path):
        np.savetxt(path + '/' + self.ensemble + '-mean', self.model_f2)
        np.savetxt(path + '/' + self.ensemble + '-err', self.model_f2_err)
        return

    def save_model_f2_mean_err(self, path, row):
        save_table_with_err(path, self.model_f2, self.model_f2_err, self.get_a(), row)

    def load_model_f2(self, path):
        self.model_f2 = np.loadtxt(path + '/' + self.ensemble + '-mean')
        self.model_f2_err = np.loadtxt(path + '/' + self.ensemble + '-err')
        return

    def plt_model_f2(self, rows, color='r', label=''):
        plt_table(
            self.model_f2, table_std=self.model_f2_err,
            unit=es.get_a(self.ensemble), rows=rows, color=color, label=label
        )
        return


def get_a0_extrapolation_list(a_list, y_list_list, yerr_list_list):
    assert len(a_list) == len(y_list_list) == len(yerr_list_list)
    x = np.array(a_list) ** 2.
    num_point = len(y_list_list[0])
    res = []
    res_err = []
    for i in range(num_point):
        y = [y_list_list[row][i] for row in range(len(y_list_list))]
        y_err = [yerr_list_list[row][i] for row in range(len(yerr_list_list))]
        y_extra, y_err_extra = ex.Extrapolate(x, y, y_err).extrapolate_to(0.)
        res.append(y_extra)
        res_err.append(y_err_extra)
    return res, res_err


def get_f2_interp(f2, x_linspace, a):
    x = np.arange(len(f2)) * a
    f2_interp = np.interp(x_linspace, x, f2)
    return f2_interp


if __name__ == '__main__':

    # plot luchang table
    LUCHANG_PATH = './data/f2/luchang/tab.txt'
    UNIT = 0.1
    R_RANGE = range(20, 21)
    luchang_table = plot_luchang_table(LUCHANG_PATH, UNIT, R_RANGE, intg='infto0', label='pion pole model', color='black')
    save_table_without_err('./out/hlbl-pion/luchang.txt', np.real(luchang_table), UNIT, 20)
    # plt.show()

    path_save_f2_jk = './ana-data/f2/jk'
    path_save_f2 = './out/hlbl-pion'
    path_save_all_config_f2 = './ana-data/f2'
    x_linspace = np.linspace(0, 10, 101)

    # ==================================================================================================================
    ensemble = 'physical-24nt96-1.0'
    # ensemble = 'heavy-24nt96-1.0'
    ana_model = DoModelAnalysis(ensemble)
    # ana_model.get_model_f2()
    # ana_model.save_model_f2_table(path_save_all_config_f2)
    ana_model.load_model_f2(path_save_all_config_f2)
    ana_model.plt_model_f2(rows=range(12, 13), label=ensemble, color='y')
    ana_model.save_model_f2_mean_err(path=path_save_f2 + '/' + ensemble + '.txt', row=12)

    f2_24_interp = get_f2_interp(ana_model.model_f2[12], x_linspace, ana_model.get_a())
    f2_err_24_interp = get_f2_interp(ana_model.model_f2_err[12], x_linspace, ana_model.get_a())
    a_24 = ana_model.get_a()

    ensemble = 'physical-32nt128-1.0'
    # ensemble = 'heavy-32nt128-1.0'
    ana_model = DoModelAnalysis(ensemble)
    # ana_model.get_model_f2()
    # ana_model.save_model_f2_table(path_save_all_config_f2)
    ana_model.load_model_f2(path_save_all_config_f2)
    ana_model.plt_model_f2(rows=range(16, 17), label=ensemble, color='g')
    ana_model.save_model_f2_mean_err(path=path_save_f2 + '/' + ensemble + '.txt', row=16)

    f2_32_interp = get_f2_interp(ana_model.model_f2[16], x_linspace, ana_model.get_a())
    f2_err_32_interp = get_f2_interp(ana_model.model_f2_err[16], x_linspace, ana_model.get_a())
    a_32 = ana_model.get_a()

    ensemble = 'physical-48nt192-1.0'
    # ensemble = 'heavy-48nt192-1.0'
    ana_model = DoModelAnalysis(ensemble)
    # ana_model.get_model_f2()
    # ana_model.save_model_f2_table(path_save_all_config_f2)
    ana_model.load_model_f2(path_save_all_config_f2)
    ana_model.plt_model_f2(rows=range(24, 25), label=ensemble, color='b')
    ana_model.save_model_f2_mean_err(path=path_save_f2 + '/' + ensemble + '.txt', row=24)

    f2_48_interp = get_f2_interp(ana_model.model_f2[24], x_linspace, ana_model.get_a())
    f2_err_48_interp = get_f2_interp(ana_model.model_f2_err[24], x_linspace, ana_model.get_a())
    a_48 = ana_model.get_a()

    ensemble = 'physical-32nt128-1.3333'
    # ensemble = 'heavy-32nt128-1.3333'
    ana_model = DoModelAnalysis(ensemble)
    # ana_model.get_model_f2()
    # ana_model.save_model_f2_table(path_save_all_config_f2)
    ana_model.load_model_f2(path_save_all_config_f2)
    ana_model.plt_model_f2(rows=range(16, 17), label=ensemble, color='r')
    ana_model.save_model_f2_mean_err(path=path_save_f2 + '/' + ensemble + '.txt', row=16)

    f2_32fine_interp = get_f2_interp(ana_model.model_f2[16], x_linspace, ana_model.get_a())
    f2_err_32fine_interp = get_f2_interp(ana_model.model_f2_err[16], x_linspace, ana_model.get_a())
    a_32fine = ana_model.get_a()

    ensemble = 'physical-48nt192-2.0'
    # ensemble = 'heavy-48nt192-2.0'
    ana_model = DoModelAnalysis(ensemble)
    # ana_model.get_model_f2()
    # ana_model.save_model_f2_table(path_save_all_config_f2)
    ana_model.load_model_f2(path_save_all_config_f2)
    ana_model.plt_model_f2(rows=range(24, 25), label=ensemble, color='c')
    ana_model.save_model_f2_mean_err(path=path_save_f2 + '/' + ensemble + '.txt', row=24)

    f2_48fine_interp = get_f2_interp(ana_model.model_f2[24], x_linspace, ana_model.get_a())
    f2_err_48fine_interp = get_f2_interp(ana_model.model_f2_err[24], x_linspace, ana_model.get_a())
    a_48fine = ana_model.get_a()

    # discretization correction
    extra_mean, extra_err = ex.get_a2_extrapolation_list(
        [a_24, a_32fine],
        [f2_24_interp, f2_32fine_interp],
        [f2_err_24_interp, f2_err_32fine_interp]
    )
    # print(extra_mean)
    plt.errorbar(x_linspace, extra_mean, yerr=extra_err, marker='*', label='discretization correction')
    save_with_err(path_save_f2 + '/model-discretization-correction.txt', x_linspace, extra_mean, extra_err)

    # finite vol
    y = extra_mean + f2_32_interp - f2_24_interp
    plt.errorbar(x_linspace, y, yerr=extra_err, marker='*', label='discretization + finite vol correction')
    save_with_err(path_save_f2 + '/model-discretization+finite-vol-correction-(24, 32fine, 32).txt', x_linspace, y, extra_err)

    plt.legend()
    plt.show()
    exit(0)

    # discretization correction
    extra_mean, extra_err = ex.get_a2_extrapolation_list(
        [a_32, a_48fine],
        [f2_32_interp, f2_48fine_interp],
        [f2_err_32_interp, f2_err_48fine_interp]
    )
    # print(extra_mean)
    plt.errorbar(x_linspace, extra_mean, yerr=extra_err, marker='*', label='discretization correction (32, 48fine)')
    save_with_err(path_save_f2 + '/model-discretization-correction-(32,48fine).txt', x_linspace, extra_mean, extra_err)

    # finite vol
    y = extra_mean + f2_48_interp - f2_32_interp
    plt.errorbar(x_linspace, y, yerr=extra_err, marker='*', label='discretization + finite vol correction (32, 48fine, 48)')
    save_with_err(path_save_f2 + '/model-discretization+finite-vol-correction-(32, 48fine, 48).txt', x_linspace, y, extra_err)

    plt.legend()
    plt.show()
    exit(0)

    # ==================================================================================================================
    # 24
    ana_24_physical_pion = Do24DPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=12)
    # ana_24_physical_pion.get_all_config_f2(1024)
    # ana_24_physical_pion.save_all_config_f2(path_save_all_config_f2)
    ana_24_physical_pion.load_all_config_f2(path_save_all_config_f2)
    ana_24_physical_pion.get_f2_jk_dict()
    ana_24_physical_pion.get_f2_jk_mean_err()
    ana_24_physical_pion.plt_f2_jk(rows=range(12, 13), color='g', label='24D_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')
    ana_24_physical_pion.save_f2_jk_mean_err(row=12, path=path_save_f2 + '/24D_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk.txt')
    # 24 interp
    ana_24_mean, ana_24_err = ana_24_physical_pion.f2_jk_mean[12], ana_24_physical_pion.f2_jk_err[12]
    ana_24_x = np.arange(len(ana_24_mean)) * ana_24_physical_pion.a
    ana_24_mean = np.interp(x_linspace, ana_24_x, ana_24_mean)
    ana_24_err = np.interp(x_linspace, ana_24_x, ana_24_err)
    print('24')
    print(ana_24_mean[0:101:5])
    print(ana_24_err[0:101:5])

    # 32
    ana_32_physical_pion = Do32DPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=16)
    # ana_32_physical_pion.get_all_config_f2(1024)
    # ana_32_physical_pion.save_all_config_f2(path_save_all_config_f2)
    ana_32_physical_pion.load_all_config_f2(path_save_all_config_f2)
    ana_32_physical_pion.get_f2_jk_dict()
    ana_32_physical_pion.get_f2_jk_mean_err()
    ana_32_physical_pion.plt_f2_jk(rows=range(16, 17), color='b', label='32D_physical_pion_xxp:16(3.11fm)_min:16(3.11fm)_ama_jk')
    ana_32_physical_pion.save_f2_jk_mean_err(row=16, path=path_save_f2 + '/32D_physical_pion_xxp:16(3.11fm)_min:16(3.11fm)_ama_jk.txt')
    # 32 interp
    ana_32_mean, ana_32_err = ana_32_physical_pion.f2_jk_mean[16], ana_32_physical_pion.f2_jk_err[16]
    ana_32_x = np.arange(len(ana_32_mean)) * ana_32_physical_pion.a
    ana_32_mean = np.interp(x_linspace, ana_32_x, ana_32_mean)
    ana_32_err = np.interp(x_linspace, ana_32_x, ana_32_err)
    print('32')
    print(ana_32_mean[0:101:5])
    print(ana_32_err[0:101:5])

    # 32 fine
    ana_32_fine_physical_pion = Do32DfinePhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=16)
    # ana_32_fine_physical_pion.get_all_config_f2(1024)
    # ana_32_fine_physical_pion.save_all_config_f2(path_save_all_config_f2)
    ana_32_fine_physical_pion.load_all_config_f2(path_save_all_config_f2)
    ana_32_fine_physical_pion.get_f2_jk_dict()
    ana_32_fine_physical_pion.get_f2_jk_mean_err()
    ana_32_fine_physical_pion.plt_f2_jk(rows=range(16, 17), color='r', label='32Dfine_physical_pion_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk')
    ana_32_fine_physical_pion.save_f2_jk_mean_err(row=16, path=path_save_f2 + '/32Dfine_physical_pion_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk.txt')
    # 32 fine interp
    ana_32_fine_mean, ana_32_fine_err = ana_32_fine_physical_pion.f2_jk_mean[16], ana_32_fine_physical_pion.f2_jk_err[16]
    ana_32_fine_x = np.arange(len(ana_32_fine_mean)) * ana_32_fine_physical_pion.a
    ana_32_fine_mean = np.interp(x_linspace, ana_32_fine_x, ana_32_fine_mean)
    ana_32_fine_err = np.interp(x_linspace, ana_32_fine_x, ana_32_fine_err)
    print('32 fine')
    print(ana_32_fine_mean[0:101:5])
    print(ana_32_fine_err[0:101:5])

    # 48
    ana_48_physical_pion = Do48IPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=24)
    # ana_48_physical_pion.get_all_config_f2(1024)
    # ana_48_physical_pion.save_all_config_f2(path_save_all_config_f2)
    ana_48_physical_pion.load_all_config_f2(path_save_all_config_f2)
    ana_48_physical_pion.get_f2_jk_dict()
    ana_48_physical_pion.get_f2_jk_mean_err()
    ana_48_physical_pion.get_jk_mean_err_save_label()
    ana_48_physical_pion.plt_f2_jk(rows=range(24, 25), color='y', label='48I_physical_pion_xxp:24(2.73fm)_min:24(2.73fm)_ama_jk')
    ana_48_physical_pion.save_f2_jk_mean_err(row=24, path=path_save_f2 + '/48I_physical_pion_xxp:24(2.73fm)_min:24(2.73fm)_ama_jk.txt')
    # 48 interp
    ana_48_mean, ana_48_err = ana_48_physical_pion.f2_jk_mean[24], ana_48_physical_pion.f2_jk_err[24]
    ana_48_x = np.arange(len(ana_48_mean)) * ana_48_physical_pion.a
    ana_48_mean = np.interp(x_linspace, ana_48_x, ana_48_mean)
    ana_48_err = np.interp(x_linspace, ana_48_x, ana_48_err)
    print('48')
    print(ana_48_mean[0:101:5])
    print(ana_48_err[0:101:5])
    exit(0)

    # discretization correction
    ana_24_mean, ana_24_err = ana_24_physical_pion.f2_jk_mean[12], ana_24_physical_pion.f2_jk_err[12]
    ana_24_x = np.arange(len(ana_24_mean)) * ana_24_physical_pion.a
    ana_32_fine_mean, ana_32_fine_err = ana_32_fine_physical_pion.f2_jk_mean[16], ana_32_fine_physical_pion.f2_jk_err[16]
    ana_32_fine_x = np.arange(len(ana_32_fine_mean)) * ana_32_fine_physical_pion.a
    # print(ana_24_mean)

    ana_24_mean = np.interp(x_linspace, ana_24_x, ana_24_mean)
    ana_24_err = np.interp(x_linspace, ana_24_x, ana_24_err)
    ana_32_fine_mean = np.interp(x_linspace, ana_32_fine_x, ana_32_fine_mean)
    ana_32_fine_err = np.interp(x_linspace, ana_32_fine_x, ana_32_fine_err)
    extra_mean, extra_err = get_a0_extrapolation_list(
        [ana_24_physical_pion.a, ana_32_fine_physical_pion.a],
        [ana_24_mean, ana_32_fine_mean],
        [ana_24_err, ana_32_fine_err]
    )
    # print(extra_mean)
    plt.errorbar(x_linspace, extra_mean, yerr=extra_err, marker='*', label='discretization correction')
    save_with_err(path_save_f2 + '/discretization correction.txt', x_linspace, extra_mean, extra_err)

    # finite vol
    ana_32_mean, ana_32_err = ana_32_physical_pion.f2_jk_mean[16], ana_32_physical_pion.f2_jk_err[16]
    ana_32_x = np.arange(len(ana_32_mean)) * ana_32_physical_pion.a
    ana_32_mean = np.interp(x_linspace, ana_32_x, ana_32_mean)
    ana_32_err = np.interp(x_linspace, ana_32_x, ana_32_err)
    plt.errorbar(x_linspace, extra_mean + ana_32_mean - ana_24_mean, yerr=extra_err, marker='*', label='discretization + finite vol correction')
    save_with_err(path_save_f2 + '/discretization + finite vol correction.txt', x_linspace, extra_mean + ana_32_mean - ana_24_mean, extra_err)

    plt.xlabel('longest distance between three points that connect to the muon line (fm)')
    plt.ylabel('f2 * 10^10 (partial integration from inf)')
    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 16
    ana_32_fine_physical_pion = Do32DfinePhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine_physical_pion.get_all_config_f2(1024)
    # ana_32_fine_physical_pion.save_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.load_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.plt_all_config_f2(rows=range(16, 17), color='r')
    ana_32_fine_physical_pion.get_f2_jk_dict()
    ana_32_fine_physical_pion.get_f2_jk_mean_err()
    ana_32_fine_physical_pion.plt_f2_jk(rows=range(16, 17), color='r', label='32Dfine_physical_pion_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk')

    xxp_limit = 24
    ana_48_physical_pion = Do48IPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_48_physical_pion.get_all_config_f2(1024)
    # ana_48.save_all_config_f2(path_save_all_config_f2)
    # ana_48.load_all_config_f2(path_save_all_config_f2)
    ana_48_physical_pion.get_f2_jk_dict()
    ana_48_physical_pion.get_f2_jk_mean_err()
    ana_48_physical_pion.get_jk_mean_err_save_label()
    ana_48_physical_pion.plt_f2_jk(rows=range(24, 25), color='y', label='48I_physical_pion_xxp:24(2.73fm)_min:24(2.73fm)_ama_jk')

    plt.legend()
    plt.show()

    exit(0)
    xxp_limit = 12
    ana_24_refine_field = Do24DRefineFieldLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_refine_field.get_all_config_f2(1024)
    # ana_24_refine_field.plt_all_config_f2(rows=range(12, 13), color='r')
    ana_24_refine_field.get_f2_jk_dict()
    ana_24_refine_field.get_f2_jk_mean_err()
    ana_24_refine_field.plt_f2_jk(rows=range(12, 13), color='g', label='24D_refine_field_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')

    xxp_limit = 12
    ana_24_physical_pion = Do24DPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_physical_pion.get_all_config_f2(1024)
    # ana_24_physical_pion.plt_all_config_f2(rows=range(12, 13), color='r')
    ana_24_physical_pion.get_f2_jk_dict()
    ana_24_physical_pion.get_f2_jk_mean_err()
    ana_24_physical_pion.plt_f2_jk(rows=range(12, 13), color='r', label='24D_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')
    plt.show()
    exit(0)

    xxp_limit = 12
    ana_24_physical_pion = Do24DPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_physical_pion.get_all_config_f2(1024)
    # ana_24_physical_pion.plt_all_config_f2(rows=range(12, 13), color='r')
    ana_24_physical_pion.get_f2_jk_dict()
    ana_24_physical_pion.get_f2_jk_mean_err()
    ana_24_physical_pion.plt_f2_jk(rows=range(12, 13), color='g', label='24D_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')

    xxp_limit = 16
    ana_32_fine_physical_pion = Do32DfinePhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine_physical_pion.get_all_config_f2(1024)
    # ana_32_fine_physical_pion.save_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.load_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.plt_all_config_f2(rows=range(16, 17), color='r')
    ana_32_fine_physical_pion.get_f2_jk_dict()
    ana_32_fine_physical_pion.get_f2_jk_mean_err()
    ana_32_fine_physical_pion.plt_f2_jk(rows=range(16, 17), color='r', label='32Dfine_physical_pion_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk')

    xxp_limit = 24
    ana_48_physical_pion = Do48IPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_48_physical_pion.get_all_config_f2(1024)
    # ana_48.save_all_config_f2(path_save_all_config_f2)
    # ana_48.load_all_config_f2(path_save_all_config_f2)
    ana_48_physical_pion.get_f2_jk_dict()
    ana_48_physical_pion.get_f2_jk_mean_err()
    ana_48_physical_pion.get_jk_mean_err_save_label()
    ana_48_physical_pion.plt_f2_jk(rows=range(24, 25), color='y', label='48I_physical_pion_xxp:24(2.73fm)_min:24(2.73fm)_ama_jk')

    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 12
    ana_24_h_physical_pion = Do24DHeavyPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    # ana_24_h_physical_pion = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_h_physical_pion.get_all_config_f2(12)
    ana_24_h_physical_pion.plt_all_config_f2(rows=range(12, 13), color='r')
    # ana_24_h_physical_pion.get_f2_jk_dict()
    # ana_24_h_physical_pion.get_f2_jk_mean_err()
    # ana_24_h_physical_pion.plt_f2_jk(rows=range(12, 13), color='b', label='24D_h_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')
    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 12
    ana_24_h_physical_pion = Do24DHeavyPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_h_physical_pion.get_all_config_f2(1024)
    # ana_24_h_physical_pion.plt_all_config_f2(rows=range(12, 13), color='r')
    ana_24_h_physical_pion.get_f2_jk_dict()
    ana_24_h_physical_pion.get_f2_jk_mean_err()
    ana_24_h_physical_pion.plt_f2_jk(rows=range(12, 13), color='b', label='24D_h_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')
    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 12
    ana_24_physical_pion = Do24DPhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24_physical_pion.get_all_config_f2(1024)
    # ana_24_physical_pion.plt_all_config_f2(rows=range(12, 13), color='r')
    ana_24_physical_pion.get_f2_jk_dict()
    ana_24_physical_pion.get_f2_jk_mean_err()
    ana_24_physical_pion.plt_f2_jk(rows=range(12, 13), color='b', label='24D_physical_pion_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')
    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 12
    ana_24 = Do24DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24.get_all_config_f2(1024)
    # ana_24.save_all_config_f2(path_save_all_config_f2)
    # ana_24.load_all_config_f2(path_save_all_config_f2)
    ana_24.get_f2_jk_dict()
    ana_24.get_f2_jk_mean_err()
    ana_24.plt_f2_jk(rows=range(12, 13), color='y', label='24D_xxp:12(2.33fm)_min:12(2.33fm)_ama_jk')

    xxp_limit = 16
    ana_32_fine_physical_pion = Do32DfinePhysicalPionLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine_physical_pion.get_all_config_f2(1024)
    # ana_32_fine_physical_pion.save_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.load_all_config_f2(path_save_all_config_f2)
    # ana_32_fine_physical_pion.plt_all_config_f2(rows=range(16, 17), color='r')
    ana_32_fine_physical_pion.get_f2_jk_dict()
    ana_32_fine_physical_pion.get_f2_jk_mean_err()
    ana_32_fine_physical_pion.plt_f2_jk(rows=range(16, 17), color='r', label='32Dfine_physical_pion_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk')

    xxp_limit = 16
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32_fine.get_all_config_f2(1024)
    # ana_32_fine.save_all_config_f2(path_save_all_config_f2)
    # ana_32_fine.load_all_config_f2(path_save_all_config_f2)
    # ana_32_fine.plt_all_config_f2(rows=range(16, 17), color='r')
    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.plt_f2_jk(rows=range(16, 17), color='g', label='32Dfine_xxp:16(2.29fm)_min:16(2.29fm)_ama_jk')

    plt.legend()
    plt.show()

    exit(0)

    xxp_limit = 12
    ana_32d = Do32DLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_32d.get_all_config_f2(1024)
    ana_32d.get_f2_mean_err()
    # ana_32d.save_f2_mean_err(path_save_f2)
    ana_32d.plt_f2(rows=range(12, 13), color='g', label='32D_xxp:12(2.33fm)_min:12(2.33fm)_ama')

    xxp_limit = 12
    ana_24d_h = Do24DHeavyLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_24d_h.get_all_config_f2(1024)
    ana_24d_h.get_f2_mean_err()
    # ana_24d_h.save_f2_mean_err(path_save_f2)
    ana_24d_h.plt_f2(rows=range(12, 13), color='b', label='24Dheavy_xxp:12(2.33fm)_min:12(2.33fm)_ama')

    xxp_limit = 24
    ana_48 = Do48ILatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    ana_48.get_all_config_f2(100)
    # ana_48.save_all_config_f2(path_save_all_config_f2)
    # ana_48.load_all_config_f2(path_save_all_config_f2)
    ana_48.get_f2_jk_dict()
    ana_48.get_f2_jk_mean_err()
    ana_48.get_jk_mean_err_save_label()
    ana_48.plt_f2_jk(rows=range(24, 25), color='y', label='48I_xxp:24(2.73fm)_min:24(2.73fm)_ama_jk')

    plt.legend()
    plt.show()

    exit(0)


    xxp_limit = 16
    ana_32_fine = Do32DfineLatticeAnalysis(mod='', ama=True, xxp_limit=xxp_limit)
    # ana_32_fine.get_all_config_f2(1024)
    # ana_32_fine.save_all_config_f2(path_save_all_config_f2)
    ana_32_fine.load_all_config_f2(path_save_all_config_f2)
    ana_32_fine.get_f2_jk_dict()
    ana_32_fine.get_f2_jk_mean_err()
    ana_32_fine.get_jk_mean_err_save_label()
    ana_32_fine.plt_f2_jk()

    plt.show()

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
