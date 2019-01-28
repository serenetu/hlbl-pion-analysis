import re
import numpy as np
import matplotlib.pyplot as plt


def old_read_table(fname, cut = 0):
    f = open(fname, 'r')
    res = []
    for line in f:
        res.append([])
        line = line[cut:]
        line = line.strip(', \n')
        comp_list = re.split(r',', line)
        last_line = res[-1]
        for comp in comp_list:
            print (comp.strip()).split()
            real, imag = (comp.strip()).split()
            last_line.append(complex(float(real), float(imag)))
    return np.array(res)


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


def plt_table(table, table_std=None, unit=1., rows=None, ylim=None, xlim=None):
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
            plt.plot(x, table[row], '*')
        else:
            plt.errorbar(x, table[row], yerr=table_std[row], marker='*')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    return


def compute_zp(l, m_pi=0.13975/1.015):
    return 1. / (2. * m_pi * l ** 3.)


def set_factor(t, l, m_pi=0.13975/1.015, mu_mass=0.1056583745 / 1.015, e=0.30282212, three=3, zw=1., zv=1.):
    q_u = 2./3.
    q_d = -1./3.
    zp = compute_zp(l, m_pi)
    fac = 2. * mu_mass * e ** 6. * three / 2. / 3. * (zv ** 4.) / (zp * zw * np.exp(-2. * m_pi * t)) * 1. / 2. * (q_u ** 2. - q_d ** 2.) ** 2.
    return fac


if __name__ == '__main__':

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

    # partial sum from 0 max_R to inf
    PATH= '/Users/tucheng/Desktop/Physics/research/hlbl-pion/hlbl-pion-analysis/test/'
    #FILE_LABEL = 'distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:30.'
    #FILE_LABEL = 'y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512.'
    #FILE_LABEL = 'distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:1024,r_pion_to_gamma:30.'
    #FILE_LABEL = 'y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512_norotate.'
    FILE_LABEL = 'pgge_rotate_from_y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096.'

    luchang_table = read_table_noimag(PATH + '/' + 'tab.txt')
    plt_table(luchang_table, unit=0.1, rows=range(20, 30))
    plt.show()
    #exit()

    table_all = []
    for i in range(1, 137):
        if i in []:
            continue
        f_name = PATH + '/' + FILE_LABEL + str(i).zfill(5)
        one_table = read_table(f_name)
        one_table = one_table.transpose()
        #one_table = de_partial_sum(one_table)
        #one_table = one_table[:, ::-1]
        one_table = partial_sum(one_table)
        print(one_table.shape)
        table_all.append(one_table)
    table_all = np.array(table_all)
    print(table_all.shape)
    factor = set_factor(t=20, l=24, zw=1.28*10.**8., zv=0.72)
    print('factor: ', factor)
    table_avg = 10**10 * factor * np.average(table_all, axis=0)
    table_std = 10**10 * factor * np.std(table_all, axis=0) * len(table_all) ** (-1./2.)
    print(table_avg.shape, table_std.shape)
    #plt_table(table_avg, table_std, ylim=(0, 15), xlim=(0, 15), unit=0.2, rows=range(10, 20))
    #plt_table(table_avg[:, ::-1], table_std[:, ::-1], unit=0.2, rows=range(1, 20))
    plt_table(table_avg, table_std, unit=0.2, rows=range(10, 20))
    plt.show()

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
