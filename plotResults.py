# script containing functions to plot results
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import pandas as pd
import seaborn as sns

lblu = "#add9f4"
lred = "#f36860"

custom_palette = [lblu, lred]


def plotDistribution(g1_ex1, g2_ex1, g1_ex2, g2_ex2, g1_ex3, g2_ex3, label_g1, label_g2, organ_dict, ylabel, save_path):
    # plot dice for all organs
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    for i in range(1, n_channels):
        organ = organs[i]

        if organ == "prostate/uterus":
            organ = "prostate or uterus"

        if (ylabel == "HD95") or (ylabel == "HD"):
            j = copy.deepcopy(i-1)
        else:
            j = copy.deepcopy(i)

        plotDistributionForOrgan(g1_ex1[:, j],
                                 g2_ex1[:, j],
                                 g1_ex2[:, j],
                                 g2_ex2[:, j],
                                 g1_ex3[:, j],
                                 g2_ex3[:, j],
                                 label_g1,
                                 label_g2,
                                 organ,
                                 ylabel,
                                 os.path.join(save_path, "{}_distribution.png".format(organ)))


def plotDistributionForOrgan(g1_ex1, g2_ex1, g1_ex2, g2_ex2, g1_ex3, g2_ex3, label_g1, label_g2, organ, ylabel, save_path):
    plt.clf()

    # Delete NaNs
    g1_ex1 = g1_ex1[~np.isnan(g1_ex1)]
    g2_ex1 = g2_ex1[~np.isnan(g2_ex1)]
    g1_ex2 = g1_ex2[~np.isnan(g1_ex2)]
    g2_ex2 = g2_ex2[~np.isnan(g2_ex2)]
    g1_ex3 = g1_ex3[~np.isnan(g1_ex3)]
    g2_ex3 = g2_ex3[~np.isnan(g2_ex3)]

    data = [g1_ex1, g2_ex1, g1_ex2, g2_ex2, g1_ex3, g2_ex3]

    labels = ['Balanced',
              '{} Training Set'.format(label_g1),
              '{} Training Set'.format(label_g2),
              'Balanced',
              '{} Training Set'.format(label_g1),
              '{} Training Set'.format(label_g2)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.2)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='',
        xlabel='',
        ylabel=ylabel,
    )

    # Now fill the boxes with desired colors
    box_colors = [lblu, lblu, lblu, lred, lred, lred]
    num_boxes = len(data)
    medians = np.empty(num_boxes)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='k', marker='*', markeredgecolor='k',
                 markersize=10)

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.0
    bottom = 0.2
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)

    # Finally, add a basic legend
    fig.text(0.80, 0.38, '{} Test Set'.format(label_g1),
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='small')
    fig.text(0.80, 0.345, '{} Test Set'.format(label_g2),
             backgroundcolor=box_colors[3],
             color='white', weight='roman', size='small')
    fig.text(0.80, 0.295, '*', color='black',
             weight='roman', size='large')
    fig.text(0.815, 0.300, ' Average Value', color='black', weight='roman',
             size='small')

    plt.axvline(x=3.5, color='k', linestyle="dashed", linewidth=1)

    if ylabel == "Volume Error":
        plt.axhline(y=0)

    plt.savefig(save_path)


def plotVolumeDistribution(vol_g1, vol_g2, label_g1, label_g2, organs_dict, save_path):
    plt.clf()
    organs = list(organs_dict.keys())

    organ_name = []
    normalised_volume = []
    group = []

    for i in range(1, len(organs)):
        organ = organs[i]

        vol_g1_i = vol_g1[:, i]
        vol_g2_i = vol_g2[:, i]

        # Get overall maximum
        vol_g2_max = np.max(vol_g2_i)
        vol_g1_max = np.max(vol_g1_i)
        v_max = np.max((vol_g1_max, vol_g2_max))

        vol_g1_i_norm = (vol_g1_i / v_max)
        vol_g2_i_norm = (vol_g2_i / v_max)

        organ_name += [organ for _ in range(vol_g1_i.shape[0])]
        organ_name += [organ for _ in range(vol_g2_i.shape[0])]

        group += [label_g1 for _ in range(vol_g1_i.shape[0])]
        group += [label_g2 for _ in range(vol_g2_i.shape[0])]

        normalised_volume += list(vol_g1_i_norm)
        normalised_volume += list(vol_g2_i_norm)

    # Now build the data frame
    df = pd.DataFrame({'Normalised Volume': normalised_volume,
                       'Group': group,
                       'Organ Name': organ_name})
    sns.boxplot(y='Normalised Volume', x='Organ Name', data=df, hue='Group', palette=custom_palette, showfliers=False)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.xlabel("")
    plt.savefig(os.path.join(save_path, "volume_norm_distribution.png"))


def plotDualDistribution(g1, g2, label_g1, label_g2, title, organs_dict, save_path):
    organs = list(organs_dict.keys())

    organ_name = []
    all_results = []
    group = []

    for i in range(1, len(organs)):
        organ = organs[i]

        if (title == "HD95") or (title == "HD"):
            j = copy.deepcopy(i-1)
        else:
            j = copy.deepcopy(i)

        g1_i = g1[:, j]
        g2_i = g2[:, j]

        g1_i = g1_i[np.isfinite(g1_i)]
        g2_i = g2_i[np.isfinite(g2_i)]

        organ_name += [organ for _ in range(g1_i.shape[0])]
        organ_name += [organ for _ in range(g2_i.shape[0])]

        group += [label_g1 for _ in range(g1_i.shape[0])]
        group += [label_g2 for _ in range(g2_i.shape[0])]

        all_results += list(g1_i)
        all_results += list(g2_i)

    # Now build the data frame
    df = pd.DataFrame({title: all_results,
                       'Group': group,
                       'Organ Name': organ_name})
    plt.clf()
    sns.boxplot(y=title, x='Organ Name', data=df, hue='Group', palette=custom_palette, showfliers=False)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.xlabel("")
    #plt.show()
    plt.savefig(save_path)