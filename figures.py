import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib2tikz import save as tikz_save


def save_foolratio_fig(X, fname, adv_dict, style_dict, legend=True):
    fig = plt.figure()
    ax = plt.subplot(111)

    inv_map = {v: k for k, v in adv_dict.items()}
    legends = []
    for ii in range(X.shape[1]):
        if X[:, ii].flatten().sum() != 0 and ii>0:
            ax.plot(X[:, 0], X[:, ii], style_dict[inv_map[ii]])
            legends.append(inv_map[ii])

            if inv_map[ii].split('-')[1] == 'pixel':
                plt.xlabel('$\epsilon$', fontsize=18)
            else:
                plt.xlabel('PSNR input (dB)', fontsize=16)

    
    plt.ylabel('PSNR output (dB)', fontsize=16)
    ax.grid()
    # plt.ylim((-5,105))
    if legend==True:
        leg = ax.legend(legends, fontsize=14)
        # set the alpha value of the legend: it will be translucent
        leg.get_frame().set_alpha(0.5)
    ax.tick_params(axis='both', labelsize='large')
    # tikz_save(fname.split('.')[0] + '.tex')
    plt.savefig( fname, format='eps', dpi=500 )


adv_dict = {
    'psnr_input': 0, 'rand-linf': 1, 'rand-l2': 2,
    'linear-linf-1': 3, 'linear-linf-10': 4, 'linear-linf-20': 5,
    'linear-l2-1': 6, 'linear-l2-10': 7, 'linear-l2-20': 8,
    'quadratic-linf-1': 9, 'quadratic-linf-10': 10,
    'quadratic-l2-1': 11, 'quadratic-l2-10': 12,
    'linear-pixel-1': 13, 'linear-pixel-100': 14,
    'quadratic-pixel-1': 15,
    'rand-pixel-1':16, 'rand-pixel-100':17
}
style_dict = {
    'rand-linf': '--ko', 'rand-l2': '--ko',
    'rand-pixel-1':'--ko', 'rand-pixel-100':'--kx',
    'linear-linf-1': '-bo', 'linear-linf-10': '-bx', 'linear-linf-20': '-bd',
    'linear-l2-1': '-bo', 'linear-l2-10': '-bx', 'linear-l2-20': '-bd',
    'linear-pixel-1': '-bo', 'linear-pixel-100': '-bx',
    'quadratic-linf-1': '-.ro', 'quadratic-linf-10': '-.rx',
    'quadratic-l2-1': '-.ro', 'quadratic-l2-10': '-.rx',
    'quadratic-pixel-1': '-.ro'
}
fname = 'figslist.txt'
os.system('ls ../summary/* > '+ fname)

with open(fname) as f:
    lines = f.readlines()
    for ii in range(0, len(lines)):
        line = lines[ii].split('\n')[0]
        X = np.loadtxt(line, delimiter=";")
        fig_name = line.split('/')[2].split('.')[0] + '.eps' 
        save_foolratio_fig(X, fig_name, adv_dict, style_dict, legend=True)
