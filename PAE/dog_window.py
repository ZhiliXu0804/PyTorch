import numpy as np
import array
import tqdm
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

import sys
sys.path.append("../")
import Library.Utility as utility
import PAE as model
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

debug = False
# Hips 0, LeftUpLeg 1, LeftLeg 2, LeftFoot 3, LeftFootSite 4, RightUpLeg 5, RightLeg 6, RightFoot 7, RightFootSite 8, Spine 9, Spine1 10, LeftShoulder 11, LeftArm 12, LeftForeArm 13, LeftHand 14, LeftHandSite 15, Neck 16, Head 17, RightShoulder 18, RightArm 19, RightForeArm 20, RightHand 21, RightHandSite 22, Tail 23, Tail1 24, Tail1Site 25
# 0, 16, 17, 18, 19, 20, 21, 22, 23, 1, 2, 6, 7, 8, 9, 10, 3, 4, 11, 12, 13, 14, 15, 24, 25, 26
kinematic_chain = [np.arange(0, 5)*3, # left leg
    np.array([0, 5, 6, 7, 8])*3, # right leg
    np.arange(10, 16)*3, # left arm
    np.array([10, 18, 19, 20, 21, 22])*3, # right arm
    np.array([16, 10, 9, 0, 23, 24, 25])*3, # spine to tail
    np.array([17, 16])*3] # head
def PrintProgress(pivot, total, resolution=5):
    if debug:
      step = max(int(total / resolution),1)
      if pivot % step == 0:
        print('Progress', round(100 * pivot / total, 2), "%")

def LoadSequences(path, debug=False, lineCount=None):
    data = []
    with open(path) as file:
        pivot = 0
        for line in file:
            pivot += 1
            PrintProgress(pivot, lineCount)
            entry = line.rstrip().split(' ')
            data.append(entry[0])
            if pivot==lineCount:
                break
    data = np.array(data, dtype=np.int64)
    return data

#binaryFile = .bin data matrix of shape samples x features
#sampleCount = number of samples in the file
#featureCount = number of features per sample
def ReadBinary(binaryFile, sampleCount, featureCount):
    print('Read binary file.')
    bytesPerLine = featureCount*4
    data = []
    with open(binaryFile, "rb") as f:
        for i in tqdm.tqdm(np.arange(sampleCount)):
            PrintProgress(i, sampleCount)
            f.seek(i*bytesPerLine)
            byte_array = f.read(bytesPerLine)
            data.append(np.float32(array.array('f', byte_array))) # Joe: convert byte array into float array
    print("")
    return np.concatenate(data).reshape(sampleCount, -1)

def plot_animation(p_init, velocity, title, isSave=False):
    positions = np.concatenate([p_init.reshape(1,-1), velocity/60], axis=0) # Joe: remember to divide velociy by 60 because it's multiplied by 60 (delta time) when export
    positions = np.cumsum(positions, axis=0)
    print(positions.shape)
    radius=4
    colors = ['red', 'blue', 'red', 'blue', 'black', 'orange']

    fig = plt.figure(figsize=(10, 10))
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([-radius / 4, radius / 4])
    ax.set_ylim3d([-radius / 4, radius / 4])
    ax.set_zlim3d([0, radius / 2])

    def update(index):
        # print(index)
        ax.lines = []
        ax.collections = []
        # ax.view_init(azim=30)
        # ax.dist = 7.5

        # # Plot root motion
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #               MAXS[2] - trajec[index, 1])
        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # Plot joints position   
        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            ax.plot3D(positions[index, chain], positions[index, chain+2], positions[index, chain+1], linewidth=2, color=color)
        
    ani = FuncAnimation(fig, update, frames=positions.shape[0], interval=1000 / 60, repeat=True)
    if isSave:
        ani.save(f"{title}.gif", fps=30)
        plt.close()
    else:
        plt.show()

def LoadBatches(sequences, velocity):
    gather = gather_window.reshape(1,-1).repeat(sequences.shape[0], 0) # Joe: mask to grab timesteps before and after current timestamp for each sequence. Shape length sequence x timewinow

    pivot = sequences.reshape(-1,1) # Joe: get current timestamp and transform to row format
    min = sequences[0]
    max = sequences[-1]

    gather = np.clip(gather + pivot, min, max) # Joe: trim anything outside of min and max boundary
    print("gather shape ", gather.shape)
    # batch = utility.ReadBatchFromMatrix(seq_one_v, gather.flatten())
    batch = velocity[gather.flatten()] # Joe: get all the velocities. shape sequence length*timewindow x joints
    print("batch shape1 ", batch.shape)
    batch = batch.reshape(gather.shape[0], gather.shape[1], -1)
    batch = batch.swapaxes(1,2)
    batch = batch.reshape(gather.shape[0], batch.shape[1]*batch.shape[2]) # Joe: shape sequence length x joints*timewindow. xxxxxyyyyyzzzz. will be transformed to [xxxx],[yyyy],[zzzz] in model.
    print("batch shape2 ", batch.shape)
    return batch

if __name__ == '__main__':
    root_path = "/home/pan/GenerativeMotion_Joe/AI4Animation-master/AI4Animation/SIGGRAPH_2022/DeepLearning/Dataset"

    shape = np.loadtxt(root_path+"/DataShape.txt", dtype=np.int64)
    sequences = LoadSequences(root_path+"/Sequences.txt", True, shape[0])
    v_filtered = ReadBinary(root_path+"/Data.bin", shape[0], shape[1])
    position = ReadBinary(root_path+"/Position.bin", shape[0], shape[1])
    v_unfiltered = ReadBinary(root_path+"/v_unfiltered.bin", shape[0], shape[1])
    
    # load trained model
    #Start Parameter Section
    window = 2.0 #time duration of the time window
    fps = 60 #fps of the motion capture data
    joints = 26 #joints of the character skeleton

    frames = int(window * fps) + 1
    input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-component of each joint)
    phase_channels = 5 #desired number of latent phase channels (usually between 2-10)
    gather_padding = (int((frames-1)/2))
    gather_window = np.arange(frames) - gather_padding

    # prepare input data
    seq_one_v = v_filtered[np.where(sequences == (1))[0], :]
    seq_one_p = position[np.where(sequences == (1))[0], :]
    seq_all = LoadBatches(np.arange(seq_one_v.shape[0]), seq_one_v) # Joe: put timewindow on each timestamp
    
    print("seq_one_v ", seq_one_v.shape)
    print("seq_all ", seq_all.shape)
    indice_velocity_pred = np.arange(0, seq_all.shape[1], frames) + gather_padding
    print("Is velocity indice correct? ", np.array_equal(seq_all[:, indice_velocity_pred], seq_one_v))

    # input animation
    plot_animation(seq_one_p[0], seq_all[:, indice_velocity_pred], 'input', True)

    # build network model
    network = torch.load("/home/pan/GenerativeMotion_Joe/AI4Animation-master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/saved_weights/10_5Channels.pt")
    network.cuda()
    network.eval()

    # run network
    seq_all = torch.from_numpy(seq_all).cuda()
    yPred, latent, signal, params = network(seq_all)
    print("yPred ",yPred.shape)

    # # output animation
    plot_animation(seq_one_p[0], yPred.cpu().detach().numpy()[:, indice_velocity_pred], 'output', True)