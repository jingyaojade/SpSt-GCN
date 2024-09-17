import pickle, logging, numpy as np
from torch.utils.data import Dataset
from dtw import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import torch


class NTU_Feeder(Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        Ad_path = '{}/{}_dynamic_edge.npy'.format(dataset_path, phase)
        try:
            self.data = np.load(data_path, mmap_mode='r')
            self.Ad = np.load(Ad_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
            print(len(self.data), len(self.Ad), len(self.label))
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        Ad_all = np.array(self.Ad[idx])
        name = self.name[idx]
        # seq_len = self.seq_len[idx]

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = self.multi_input(data[:,:self.T,:,:])

        data_new = []  # joint, velocity, bone
        Ad = np.zeros((data.shape[2], data.shape[2]))
        # Ad_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
            Ad = -Ad_all[0] + np.identity(data.shape[2])
            # Ad_new.append(Ad)
        if 'V' in self.inputs:
            data_new.append(velocity)
            Ad = -Ad_all[1] + np.identity(data.shape[2])
            # Ad_new.append(Ad)
        if 'B' in self.inputs:
            data_new.append(bone)
            Ad = -Ad_all[2] + np.identity(data.shape[2])
            # Ad_new.append(Ad)
        data_new = np.stack(data_new, axis=0)
        # Ad_new = np.array(Ad_new)

        return data_new, label, Ad, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone


class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location
