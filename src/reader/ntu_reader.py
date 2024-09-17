import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# import sys
# sys.path.append("D:/publication1/ss_GCN/src/dataset/graphs")
# import src.dataset.graphs


class NTU_Reader():
    def __init__(self, args, connect_joint, root_folder, transform, ntu60_path, ntu120_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.conn = connect_joint
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        try:
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        except:
            logging.info('')
            logging.error('Error: Wrong in loading ignored sample file {}'.format(ntu_ignored))
            raise ValueError()

        # Get skeleton file list
        self.file_list = []
        for folder in [ntu60_path, ntu120_path]:
            for filename in os.listdir(folder):
                self.file_list.append((folder, filename))
            if '120' not in self.dataset:  # for NTU 60, only one folder
                break

    def read_file(self, file_path):
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    person_info = fr.readline().strip().split()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
        return skeleton[:,:frame_num,:,:], frame_num

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def gendata(self, phase):
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        sample_Ad = []
        sample_j_v_b = []
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        for folder, filename in iterizer:
            if filename in self.ignored_samples:
                continue

            # Get sample information
            file_path = os.path.join(folder, filename)
            setup_loc = filename.find('S')
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            action_class = int(filename[(action_loc+1):(action_loc+4)])

            # Distinguish train or eval sample
            if self.dataset == 'ntu-xview':
                is_training_sample = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120':
                is_training_sample = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120':
                is_training_sample = (setup_id in self.training_sample)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            skeleton, frame_num = self.read_file(file_path)

            # Select person by max energy
            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)
            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)

        # Save label
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)

        # Save data
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # calculate dynamic edge matrix
        N, C, T, V, M = sample_data.shape
        for sample in tqdm(range(N)):
            data = sample_data[sample, :, :, :, :]
            # multi input
            joint, velocity, bone = self.multi_input(data)

            # dynamic edge
            Ad = self.dynamic_edge(data, joint, velocity, bone)

            # sample_j_v_b.append(data_new)
            sample_Ad.append(Ad)

            # Save data
        np.save('{}/{}_dynamic_edge.npy'.format(self.out_path, phase), sample_Ad)


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


    def dynamic_edge(self, data, joint, velocity, bone):
        C, T, V, M = data.shape
        Ad_joint = np.zeros((V, V))
        Ad_velocity = np.zeros((V, V))
        Ad_bone = np.zeros((V, V))
        Ad = []
        margin_joint = [3,15,19,21,23]
        for vertex in margin_joint:
            for neighbor in [3,15,19,21,23]:
                if vertex == neighbor:
                    pass
                else:
                    joint_vertex = joint[:, :, vertex, :].reshape(T, 2*C*M)
                    joint_neighbor = joint[:, :, neighbor, :].reshape(T, 2*C*M)
                    Ad_joint[vertex, neighbor], _ = fastdtw(joint_vertex, joint_neighbor, dist=euclidean)

                    velocity_vertex = velocity[:, :, vertex, :].reshape(T, 2 * C * M)
                    velocity_neighbor = velocity[:, :, neighbor, :].reshape(T, 2 * C * M)
                    Ad_velocity[vertex, neighbor], _ = fastdtw(velocity_vertex, velocity_neighbor, dist=euclidean)

                    bone_vertex = bone[:, :, vertex, :].reshape(T, 2 * C * M)
                    bone_neighbor = bone[:, :, neighbor, :].reshape(T, 2 * C * M)
                    Ad_bone[vertex, neighbor], _ = fastdtw(bone_vertex, bone_neighbor, dist=euclidean)

        Ad.append(np.linalg.pinv(Ad_joint))
        Ad.append(np.linalg.pinv(Ad_velocity))
        Ad.append(np.linalg.pinv(Ad_bone))
        return np.array(Ad)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
