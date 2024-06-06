
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio as iio


def reshape_npy_array(file):
    """reshape the array before plotting it"""
    data = np.load(file)
    reshaped_data = np.reshape(data, (748,54))
    return reshaped_data
#     np.save('reshaped_sample.npy',reshaped_data)
# reshape_npy_array('sample.npy')
#######################################################################################################################################

def create_directory(directory_name):
    '''create a directory for the png files and the gif file'''
    try:
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")

#######################################################################################################################################

def plot_limbs(ax, jointsX, jointsY, jointsZ):
    '''plot_limbs function tries to draw links between joints.
    Must check that the joint pairs  is correct.'''
    joint_pairs = [(3,4),(4,5),(5,6),(12,13),(11,12),(0,11),(0,14),(14,15),(15,16),(7,8),(8,9),(9,10),(17,7),(2,1),(1,0),(3,17),(2,17)
    ]
    for joint_pair in joint_pairs:
        indexes = list(joint_pair)
        ax.plot(np.take(jointsX, indexes), np.take(jointsY, indexes), np.take(jointsZ, indexes), linewidth=2, color='black')

#######################################################################################################################################

def plot_skel(x,output_directory,title):
    '''this function allows to plot the nodes and links
     of the skeleton as a lot of png files and then gather them in a gif.'''
    # x=reshape_npy_array(x)
    seq=x[0]

    out_root_dir = output_directory
    out_pngs_dir = out_root_dir + 'pngs/'
    out_gifs_dir = out_root_dir 
    create_directory(out_pngs_dir)
 
    create_directory(out_gifs_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=23., azim=-110)
    file_names = []
    numJoints = 18
    dim = 3
    for i in range(0, seq.shape[0]):
        plt.cla()
        skelX = seq[i].reshape(numJoints, dim)[:, 0]
        skelY = seq[i].reshape(numJoints, dim)[:, 2]
        skelZ = seq[i].reshape(numJoints, dim)[:, 1]
        ax.scatter(skelX, skelY, skelZ, c='green',depthshade=False)
        # plot limbs
        plot_limbs(ax, skelX, skelY, skelZ)
        ax.set_xlim(-2, 2)
        ax.set_zlim(-1, 1)
        ax.set_ylim(0, 5)
        curr_file_name = str(i) + '.png'
        file_names.append(curr_file_name)
        plt.savefig(out_pngs_dir + curr_file_name)
        images = []
    plt.close()
    for file_name in file_names:
        images.append(iio.imread(out_pngs_dir + file_name))

    out_file = out_gifs_dir +title+'.gif'
    kargs = {'duration': 0.01}
    iio.mimsave(out_file, images, 'GIF', **kargs)






