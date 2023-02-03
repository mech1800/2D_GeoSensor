import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 0～9まで
for number in [0,1,2,3,4,5,6,7,8,9]:
    # datasetをloadする
    pre_geometry_dataset = np.load('dataset/'+str(number)+'/pre_geometry.npy')
    geometry_dataset = np.load('dataset/'+str(number)+'/geometry.npy')
    contact_dataset = np.load('dataset/'+str(number)+'/contact.npy')
    stress_dataset = np.load('dataset/'+str(number)+'/stress.npy')
    force_dataset = np.load('dataset/'+str(number)+'/force.npy')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(pre_geometry_dataset[400], cmap=cm.jet)
    ax.set_xticks([])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(geometry_dataset[400], cmap=cm.jet)
    ax.set_xticks([])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(contact_dataset[400], cmap=cm.jet)
    ax.set_xticks([])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(stress_dataset[400], cmap=cm.jet)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax)
    cbar.ax.tick_params(labelsize=5)
    ax.set_xticks([])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(force_dataset[400], cmap=cm.jet, vmin=0, vmax=30)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_ylim(0, 30)
    ax.set_xticks([])
    plt.show()