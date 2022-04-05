import open3d as o3d

import torch
import numpy as np
from os import listdir
import tqdm
from pathlib import Path
import bl
device = torch.device('cuda')


def test_chairs(num_exp, step):

    chair_dict = {}

    mean, std = torch.load('betas_params.pt')

    files = []
    betas = []

    # Name of folder with untrained gen model chairs
    folder = 'experiments/chair/0'
    files += itemsdir(folder, '.pt')
    # Generate random betas for unconditioned chairs
    betas.append((mean + torch.randn((100, len(std)))
                 * std).float().to(device))

    for i in range(num_exp):
        x = (i+1)*step
        # Load in folder for chairs output at epoch x
        folder = f'experiments/chair/{x}'
        files += itemsdir(folder, '.pt')

    bsz = 100

    for i in tqdm.tqdm(range(0, len(files), bsz)):

        chairs = []
        keys = []
        curr_betas = []

        for j in range(min(len(files) - i, bsz)):
            key = files[i+j]
            item = torch.load(key)
            # Load in input for file
            chair = item['pcd']
            beta = item['beta']

            chairs.append(chair)
            keys.append(key)
            curr_betas.append(beta)
        curr_betas = torch.stack(curr_betas)

        poses = bl.simulation(chairs)
        poses = bl.more_pose(
            chairs, torch.tensor(poses), betas=curr_betas)
        losses = bl.pose_to_loss_v2(poses, chairs, betas=curr_betas)

        for (key, loss, pose, b) in list(zip(keys, losses, poses, curr_betas)):
            chair_dict[key] = {
                'loss': loss,
                'pose': pose,
                'betas': b.cpu()
            }

    # Save chair dict
    return chair_dict


def plot_results(num_exp, step, chair_dict):

    values = [[] for i in range(num_exp+1)]
    for key in chair_dict:

        val = d[key]['loss']

        # Add value to correct bin based on ascending order of epoch

    vals = np.array(vals)

    histogram(vals)
    cdf(vals)


def itemsdir(name, ends='.txt'):

    if ends is None:
        return [f'{name}/{f}' for f in listdir(name)]
    return [f'{name}/{f}' for f in listdir(name) if f.endswith(ends)]


def histogram(vals, rows=1, cols=1, rang=None, titles=None, ymax=None):

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=rows, ncols=cols, figure=fig)
    axs = gs.subplots()

    if isinstance(vals[0], float):
        vals = np.array(vals).reshape(1, -1)
        v = vals
    else:
        v = np.concatenate(vals)
    if rang is None:
        mini, maxi = np.min(v), np.max(v)
    else:
        mini, maxi = rang

    hists = []
    hist_max = 0.

    for i in range(len(vals)):
        hist, edges = np.histogram(vals[i], range=(mini, maxi))
        hist = hist / len(vals[i])

        hist_max = max(max(hist), hist_max)
        hists.append(hist)

    if ymax is None:
        lim = hist_max * 1.01
    else:
        lim = ymax
    edges = edges[:-1]
    width = edges[1] - edges[0]

    for i in range(len(vals)):
        if rows == 1 and cols == 1:
            ax = axs
        elif rows == 1 or cols == 1:
            ax = axs[i]
        else:
            ax = axs[i//cols, i % cols]

        ax.bar(edges, hists[i], width, align='edge')
        if titles is not None:
            ax.title.set_text(titles[i])
        ax.set_ylim([0, lim])

    fig.savefig('histo.png')


def cdf(vals, rang=None, titles=None):

    if isinstance(vals[0], float):
        vals = np.array(vals).reshape(1, -1)
        v = vals
    else:
        v = np.concatenate(vals)
    if rang is None:
        mini, maxi = np.min(v), np.max(v)
    else:
        mini, maxi = rang

    plt.xlim(mini, maxi)
    p = np.arange(len(vals[0])) / (len(vals[0]) - 1)
    hs = []
    for i in range(len(vals)):
        h, = plt.plot(np.sort(vals[i]), p)
        hs.append(h)
    if titles is not None:
        plt.legend(hs, titles)

    plt.show()
    plt.savefig("plot.png")


if __name__ == "__main__":

    # Number of folders to iterate over
    num_exp = 49
    # Step between epochs
    step = 2000

    chair_dict = test_chairs(num_exp, step)
    plot_results(num_exp, step, chair_dict)
