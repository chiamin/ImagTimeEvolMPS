import matplotlib.pyplot as plt
import math
import numpy as np
import analysis as ana

def get_default_colors ():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors

def plot_square_latt (ax, lx, ly, **args):
    pltargs = dict(c='gray',zorder=0)
    pltargs.update (args)
    linesx, linesy = [],[]
    for y in range(1,ly+1):
        ax.plot ([1,lx],[y,y],**pltargs)
    for x in range(1,lx+1):
        ax.plot ([x,x],[1,ly],**pltargs)

def round_n (num, n):
    if num != 0:
        a = round(num, -int(math.floor(math.log10(abs(num))) - (n - 1)))
        if abs(a) < 1e-2:
            a = '{:.0e}'.format(a).replace('e-0','e-')
        return a
    else:
        return 0

def rescale_value (x, xmax, plotmax, xmin=0, plotmin=0):
# Scale base on xmax.
# If x == xmax, return plotmax
    #return x * plotmax / xmax
    re = (x-xmin)*(plotmax-plotmin)/(xmax-xmin) + plotmin
    return re

def plot_arrow (ax, x, y, length, c):
    halfl = 0.5*length
    ax.arrow (x, y-halfl, dx=0, dy=2*halfl, width=0.05, length_includes_head=True, head_width=0.2, head_length=0.5*abs(halfl), fc=c, ec=c, capstyle='butt')
    #prop = dict(arrowstyle="->", facecolor=c)
    #ax.annotate("", xy=(x, y+halfl), xytext=(x, y-halfl), arrowprops=prop)

def plot_szi (ax, x, y, sz, szmax=0.5, arrow_max=1, AFDomain=False, szcutoff=1e-3):
    color1 = 'darkorange'
    color2 = 'royalblue'
    if abs(sz) > szcutoff:
        arrow_size = rescale_value (sz, szmax, arrow_max)
        if AFDomain:
            if (x % 2 == y % 2) == (sz > 0): arrow_color = color1
            else:                            arrow_color = color2
        else:
            if sz > 0: arrow_color = color1
            else:      arrow_color = color2
        plot_arrow (ax, x, y, arrow_size, arrow_color)

def plot_circle (ax, x, y, h, hmax=1, hmin=0, circle_max=0.3, valcutoff=1e-3, color1='limegreen', color2='r', **args):
    if abs(h) > valcutoff:
        circle_size = rescale_value (h, hmax, circle_max, xmin=hmin)
        if h > 0: args['color'] = color1
        else:     args['color'] = color2
        if 'ec' not in args:
            args['ec'] = 'k'
        p = plt.Circle((x, y), circle_size, **args)
        ax.add_artist (p)

def plot_bond (ax, x1, y1, x2, y2, val, valmax=0.5, valmin=0., width_max=8, width_min=0, valcutoff=1e-3, text_dict=dict(), plot_dict=dict()):
    if abs(val) > valcutoff:
        if 'lw' not in plot_dict:
            width = rescale_value (val, valmax, width_max, xmin=valmin, plotmin=width_min)
            width = abs(width)
            plot_dict['lw'] = width
        if 'solid_capstyle' not in plot_dict:
            plot_dict['solid_capstyle'] = 'round'
        if 'zorder' not in plot_dict:
            plot_dict['zorder'] = 1
        if 'ls' not in plot_dict:
            if val > 0:
                plot_dict['ls'] = '-'
            else:
                plot_dict['ls'] = '--'
                plot_dict['dashes'] = [2,1]
        ax.plot ([x1,x2], [y1,y2], **plot_dict)

    if len(text_dict) != 0:
        textx = text_dict.get('textx',[])
        texty = text_dict.get('texty',[])
        if type(textx) != list:
            textx = [textx]
        if type(texty) != list:
            texty = [texty]
        if (x1 == x2 and x1 in textx) or (y1 == y2 and y1 in texty):
            x = (x1+x2)*0.5
            y = (y1+y2)*0.5
            xoffset = text_dict.get('xoffset',0)
            yoffset = text_dict.get('yoffset',0)
            fontsize = text_dict.get('fontsize',20)
            rounddig = text_dict.get('rounddig',2)
            color = text_dict.get('color','tab:orange')
            if abs(val) >= valcutoff:
                ax.text (x+xoffset, y+yoffset, round_n(val,rounddig), horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color=color)

def plot_sz (ax, xs, ys, szs, AFDomain=False, fontsize=16):
    szmax = max(szs)
    lx,ly = max(xs),max(ys)
    for x, y, sz in zip(xs, ys, szs):
        plot_szi (ax, x, y, sz, szmax=szmax, AFDomain=AFDomain)
        # print numbers
        if y == ly:
            ax.text (x, ly+1, round_n(sz,2), horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    ax.text (lx/2, ly+1.4, 'magnetic moment', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

def plot_h (ax, xs, ys, hs, fontsize=16):
    hmax = max(hs)
    lx,ly = max(xs),max(ys)
    for x, y, h in zip(xs, ys, hs):
        plot_circle (ax, x, y, h, hmax=hmax)
        # print numbers
        if y == 1:
            ax.text (x, 0, round_n(h,2), horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    ax.text (lx/2, -0.4, 'hole density', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

def plot_hsz (ax, xs, ys, hs, szs, AFDomain=False, fontsize=16):
    hmax, szmax = max(hs), max(szs)
    lx,ly = max(xs),max(ys)
    for x, y, h, sz in zip(xs, ys, hs, szs):
        plot_circle (ax, x, y, h, hmax=hmax)
        plot_szi (ax, x, y, sz, szmax=szmax, AFDomain=AFDomain)
        # print numbers
        if y == 1:
            ax.text (x, 0, round_n(h,2), horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
        if y == ly:
            ax.text (x, ly+1, round_n(sz,2), horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    ax.text (lx/2, -0.4, 'hole density', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    ax.text (lx/2, ly+1.4, 'magnetic moment', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

def plot_delta (ax, bonds, deltas, lx, ly, fontsize=16):
    delta_max = max(deltas)
    for bond,delta in zip(bonds,deltas):
        x1,y1,x2,y2 = bond
        if delta > 0: c = 'tab:blue'
        else: c = 'tab:red'
        if y1 == 1 and y2 == ly:
            plot_bond (ax, x1, y1, x2, y1-0.5, delta, valmax=delta_max, plot_dict={'c':c})
            plot_bond (ax, x2, y2, x2, y2+0.5, delta, valmax=delta_max, plot_dict={'c':c})
        else:
            plot_bond (ax, x1, y1, x2, y2, delta, valmax=delta_max, plot_dict={'c':c})
        if y1 == 1 and y2 == 2:
            ax.text (x1, -1, round_n(delta,2), horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    ax.text (lx/2, -1.4, 'vertical $\Delta$', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

def gen_figure (lx, ly, marginx=0, marginy=0):
    f,ax = plt.subplots (figsize=(lx+marginx,ly+marginy))
    ax.set_axis_off()
    ax.set_aspect('equal', 'datalim')
    return f, ax

def set_figure (f, ax):
    f.tight_layout (pad=0, h_pad=0, w_pad=0)

def add_labels (ax, lx, ly, *info, fontsize=24):
    x, y = lx/2, 0
    label = info[0]
    nline = 1
    for i in range(1,len(info)):
        if i % 2 == 0:
            label += '\n'
        else:
            label += ',  '
        label += info[i]
        nline += 1
    ax.text (x, y, label, horizontalalignment='center', verticalalignment='top', fontsize=fontsize, bbox=dict(boxstyle='square',facecolor='None'))

    # adjust figure size
    fig = plt.gcf()
    fsize = fig.get_size_inches()
    fsize[1] += nline * 0.4
    fig.set_size_inches (fsize)

    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()

def write_val_x (ax, xs, vals, y, itv=1, **args):
    xs = xs[::itv]
    vals = vals[::itv]
    for x,val in zip(xs, vals):
        ax.text (x, y, round_n(val,2), horizontalalignment='center', verticalalignment='center', **args)

def get_xys (lx, ly):
    xs, ys = [],[]
    for x in range(1,lx+1):
        for y in range(1,ly+1):
            xs.append (x)
            ys.append (y)
    return xs, ys

if __name__ == "__main__":
    fnames = ["ntau0.dat","ntau10.dat","ntau20.dat","ntau30.dat","ntau40.dat","ntau50.dat","ntau60.dat","ntau70.dat","ntau80.dat"]

    lx,ly = 2,2
    s = 2
    for fil in fnames:
        data = ana.read_monte_carlo_file(fil)
        obs = ana.compute_mean_and_error_with_sign(data)
        nup = obs["nup"][0]
        ndn = obs["ndn"][0]

        print(sum(nup), sum(ndn))

        f,ax = gen_figure (s*lx, s*ly)

        plot_square_latt (ax, lx, ly)

        xs, ys, hs, szs = [],[],[],[]
        i = 0
        for x in range(1,lx+1):
            for y in range(1,ly+1):
                xs.append(x)
                ys.append(y)
                hs.append(1 - nup[i] - ndn[i])
                szs.append(nup[i] - ndn[i])
                i += 1
        plot_hsz (ax, xs, ys, hs, szs)
        plt.tight_layout()
        f.savefig(fil.replace(".dat","_hsz.pdf"))
        #plt.show()

