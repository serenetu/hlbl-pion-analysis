
def get_vals_in_line(line):
    res = ''
    line_fix = line.replace('[0] ', '')
    line_fix = line_fix.strip()
    if ',' in line_fix:
        vals = line_fix[:-1].split(',')
        for val in vals:
            try:
                real, img = val.strip().split()
                float(real)
                float(img)
            except ValueError:
                return False
            else:
                real, img = val.strip().split()
                res += real + ' ' + img + ' '
    else:
        vals = line_fix.split()
        for val in vals:
            try:
                float(val)
            except ValueError:
                return False
            else:
                res += val + ' '
    return res[:-1]


def get_f_out_name(line):
    res = line.split()[-1].split('/')[-1]
    if '[0]' in res:
        res = res[:-3]
    return res


def clean(f_name, out_path):
    f = open(f_name, 'r')
    if_write = False
    for line in f:
        nums_out_files = 0
        nums_out_lines = 0

        if 'Compute Pair' in line:
            f_out_name = get_f_out_name(line)

        if 'Complex_Table' in line:
            if_write = True
            f_out = open(out_path + '/' + f_out_name, 'w')
            continue

        if if_write == True:
            vals = get_vals_in_line(line)
            if vals == False:
                nums_out_lines = 0
                f_out.close()
                nums_out_files += 1
                print('creat ' + f_out_name + ' in ' + out_path)
                if_write = False
                continue
            f_out.write(vals + '\n')
            nums_out_lines += 1
    f.close()
    return


def clean_onepair(f_name, out_path, out_label):
    f = open(f_name, 'r')
    if_write = False
    num = 0
    for line in f:
        nums_out_files = 0
        nums_out_lines = 0

        if 'Complex_Table' in line:
            num += 1
            if_write = True
            f_out = open(out_path + '/' + out_label + str(num).zfill(5), 'w')
            continue

        if if_write == True:
            vals = get_vals_in_line(line)
            if vals == False:
                nums_out_lines = 0
                f_out.close()
                nums_out_files += 1
                print('creat ' + out_label + str(num).zfill(5) + ' in ' + out_path)
                if_write = False
                continue
            f_out.write(vals + '\n')
            nums_out_lines += 1
    f.close()
    return


if __name__ == '__main__':
    f_name = './test/2019.01.28-17:04:42-41854'
    out_path = './test/'
    out_label = 'pgge_rotate_from_y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096.'
    clean_onepair(f_name, out_path, out_label)