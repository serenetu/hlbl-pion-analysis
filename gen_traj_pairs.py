import os
import random


class GenTrajPairs(object):

    def __init__(self):
        self.num_pairs = 10000
        self.step = 10
        self.sep_limit = 50
        self.save_path = './TrajPairs'
        self.pairs = []

        self.ensemble = None

        self.start = None
        self.end = None

        self.file_name = None
        return

    def get_file_name(self):
        file_name = 'ensemble:{}_start:{}_end:{}_step:{}_numpairs:{}_seplimit:{}'.format(
            self.ensemble,
            self.start,
            self.end,
            self.step,
            self.num_pairs,
            self.sep_limit
        )
        return file_name

    def gen_n_pairs(self):
        traj_range = range(self.start, self.end + 1, self.step)
        while True:
            traj1 = random.choice(traj_range)
            traj2 = random.choice(traj_range)
            if abs(traj1 - traj2) >= self.sep_limit:
                self.pairs.append((traj1, traj2))
            if len(self.pairs) == self.num_pairs:
                break
        return

    def save_pairs(self):
        filename = self.save_path + '/' + self.file_name
        f = open(filename, 'w')
        for one_pair in self.pairs:
            f.write('{} {}\n'.format(one_pair[0], one_pair[1]))
        f.close()
        return


class Gen24DTrajPairs(GenTrajPairs):

    def __init__(self):
        super(Gen24DTrajPairs, self).__init__()
        self.ensemble = '24D-0.00107'

        self.start = 1000
        self.end = 3000

        self.file_name = self.get_file_name()
        return


class Gen24DHeavyTrajPairs(GenTrajPairs):

    def __init__(self):
        super(Gen24DHeavyTrajPairs, self).__init__()
        self.ensemble = '24D-0.0174'

        self.start = 200
        self.end = 1000

        self.file_name = self.get_file_name()
        return


class Gen32DTrajPairs(GenTrajPairs):

    def __init__(self):
        super(Gen32DTrajPairs, self).__init__()
        self.ensemble = '32D-0.00107'

        self.start = 680
        self.end = 2000

        self.file_name = self.get_file_name()
        return


class Gen32DFineTrajPairs(GenTrajPairs):

    def __init__(self):
        super(Gen32DFineTrajPairs, self).__init__()
        self.ensemble = '32Dfine-0.0001'

        self.start = 200
        self.end = 2000

        self.file_name = self.get_file_name()
        return


class Gen48ITrajPairs(GenTrajPairs):

    def __init__(self):
        super(Gen48ITrajPairs, self).__init__()
        self.ensemble = '48I-0.00078'

        self.start = 500
        self.end = 3000

        self.file_name = self.get_file_name()
        return


if __name__ == '__main__':
    # 24D
    gen_traj_pairs = Gen24DTrajPairs()
    gen_traj_pairs.gen_n_pairs()
    gen_traj_pairs.save_pairs()
    # 24DHeavy
    gen_traj_pairs = Gen24DHeavyTrajPairs()
    gen_traj_pairs.gen_n_pairs()
    gen_traj_pairs.save_pairs()
    # 32D
    gen_traj_pairs = Gen32DTrajPairs()
    gen_traj_pairs.gen_n_pairs()
    gen_traj_pairs.save_pairs()
    # 32DFine
    gen_traj_pairs = Gen32DFineTrajPairs()
    gen_traj_pairs.gen_n_pairs()
    gen_traj_pairs.save_pairs()
    # 48I
    gen_traj_pairs = Gen48ITrajPairs()
    gen_traj_pairs.gen_n_pairs()
    gen_traj_pairs.save_pairs()
