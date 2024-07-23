import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time
def plot_3d_motion(x, mp_joints, kinematic_tree, title='sample', figsize=(10, 10), fps=30, radius=10):
    matplotlib.use('Agg')

    # Define initial limits of the plot
    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.grid(b=False)

    # Function to plot a floor in the animation
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # Create the figure and axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()

    def update(index):
        """
        Update function for the matplotlib animation
        """
        # Update the progress bar
        bar.update(1)
        # Clear the axis and setting initial parameters
        ax.clear()
        plt.axis('off')
        ax.view_init(elev=70, azim=-90)
        ax.dist = 7.5
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        mp_data = []
        for joints in mp_joints:
            data = joints.copy().reshape(len(joints), -1, 3)
            MINS = data.min(axis=0).min(axis=0)
            height_offset = MINS[1]
            data[:, :, 1] -= height_offset
            mp_data.append({"joints": data})

        for pid, data in enumerate(mp_data):
            for i, chain in enumerate(kinematic_tree):
                linewidth = 3.0
                ax.plot3D(data["joints"][index, chain, 0],
                          data["joints"][index, chain, 1],
                          data["joints"][index, chain, 2],
                          linewidth=linewidth,
                          color='blue',  # Replace with color logic if needed
                          alpha=1)

    # Generate animation
    frame_number = 700
    bar = tqdm(total=frame_number + 1, disable=True)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=True)
    ani.save('../sample.gif', fps=fps)
    plt.close()

joint_pairs = [
    (2, 1), (1, 0), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (0, 11), (0, 14), 
    (12, 13), (11, 12), (14, 15), (15, 16), (3, 17), (2, 17), (7, 17),
]

sample = np.load('../data/Kimore/data.npy')
mp_joint = []
x = sample[0:]
joint = x.reshape(-1, 18, 3)
mp_joint.append(joint)


start = time.time()
plot_3d_motion(joint_pairs, mp_joint, joint_pairs, fps=30)
end = time.time()
print(end - start)





# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import imageio as iio


# def reshape_npy_array(file):
#     """reshape the array before plotting it"""
#     data = np.load(file)
#     reshaped_data = np.reshape(data, (748,54))
#     return reshaped_data
# #     np.save('reshaped_sample.npy',reshaped_data)
# # reshape_npy_array('sample.npy')
# #######################################################################################################################################

# def create_directory(directory_name):
#     '''create a directory for the png files and the gif file'''
#     try:
#         os.makedirs(directory_name)
#         print(f"Directory '{directory_name}' created successfully.")
#     except FileExistsError:
#         print(f"Directory '{directory_name}' already exists.")

# #######################################################################################################################################

# def plot_limbs(ax, jointsX, jointsY, jointsZ):
#     '''plot_limbs function tries to draw links between joints.
#     Must check that the joint pairs  is correct.'''
#     joint_pairs = [(3,4),(4,5),(5,6),(12,13),(11,12),(0,11),(0,14),(14,15),(15,16),(7,8),(8,9),(9,10),(17,7),(2,1),(1,0),(3,17),(2,17)
#     ]
#     for joint_pair in joint_pairs:
#         indexes = list(joint_pair)
#         ax.plot(np.take(jointsX, indexes), np.take(jointsY, indexes), np.take(jointsZ, indexes), linewidth=2, color='black')

# #######################################################################################################################################

# def plot_skel(x,output_directory,title):
#     '''this function allows to plot the nodes and links
#      of the skeleton as a lot of png files and then gather them in a gif.'''
#     # x=reshape_npy_array(x)
#     seq=x[0]

#     out_root_dir = output_directory
#     out_pngs_dir = out_root_dir + 'pngs/'
#     out_gifs_dir = out_root_dir 
#     create_directory(out_pngs_dir)
 
#     create_directory(out_gifs_dir)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=23., azim=-110)
#     file_names = []
#     numJoints = 18
#     dim = 3
#     for i in range(0, 500):
#         plt.cla()
#         skelX = seq[i].reshape(numJoints, dim)[:, 0]
#         skelY = seq[i].reshape(numJoints, dim)[:, 2]
#         skelZ = seq[i].reshape(numJoints, dim)[:, 1]
#         ax.scatter(skelX, skelY, skelZ, c='green',depthshade=False)
#         # plot limbs
#         plot_limbs(ax, skelX, skelY, skelZ)
#         ax.set_xlim(-2, 2)
#         ax.set_zlim(-1, 1)
#         ax.set_ylim(0, 5)
#         curr_file_name = str(i) + '.png'
#         file_names.append(curr_file_name)
#         plt.savefig(out_pngs_dir + curr_file_name)
#         images = []
#     plt.close()
#     for file_name in file_names:
#         images.append(iio.imread(out_pngs_dir + file_name))

#     out_file = out_gifs_dir +title+'.gif'
#     kargs = {'duration': 0.01}
#     iio.mimsave(out_file, images, 'GIF', **kargs)






