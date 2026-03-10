import os
import ast
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from utils import sort_nicely, angle_difference
import pandas as pd
import copy
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, deque
from torch.utils.data import Sampler


def read_and_concat(filename):
    if isinstance(filename, pd.DataFrame):
        df = filename
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename, engine="openpyxl")
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    results = []
    for i, row in df.iterrows():
        id_name = 'id' if 'id' in row else 'ID'
        results.append(f"{row[id_name]}_{row['label']}")
    return results


class Trajectory(Dataset):
    def __init__(self, traj_root):
        self.traj_root = traj_root
        traj_files = os.listdir(traj_root)
        traj_files = [t for t in traj_files if t.endswith('npy')]
        self.traj_files = sort_nicely(traj_files)
        self.trajs = []
        for t in self.traj_files:
            traj = np.load(os.path.join(traj_root, t), allow_pickle=True).item()
            self.trajs.append(traj)

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        name = self.traj_files[idx]

        avoider = np.array(traj['avoider'])
        avoidee = np.array(traj['avoidee'])
        avoider[:, :-1] /= 10
        avoidee[:, :-1] /= 10
        avoider[:, 1] -= 1
        avoidee[:, 1] -= 1
        relative = avoidee - avoider
        relative[:, -1] = angle_difference(avoidee[:, -1], avoider[:, -1]) / 180
        avoider[:, -1] = (avoider[:, -1])

        avoider[:, -1] = (avoider[:, -1] - 180) / 180
        avoidee[:, -1] = (avoidee[:, -1] - 180) / 180

        data = np.hstack([avoider, relative])
        data = torch.from_numpy(data[75:105]).float().unsqueeze(0)
        return data, int(name.replace('.npy', ''))


class Charade(Dataset):
    def __init__(self, filename, force=False, norm=True, test_file=None, training=True, infering=False, use_relative=False, reverse=False, infer_file=None, input_format = "excel"): #YY
        self.filename = filename
        if input_format == "excel":
            data = pd.read_excel(filename, engine="openpyxl")
        elif input_format == "pkl":
            data = pd.read_pickle(filename)
        self.fps = 50
        self.norm = norm
        self.use_relative = use_relative
        self.reverse = reverse #YY
        self.id_label_mapping = {}

        usable = data['usable']

        self.data = []
        for i in range(len(usable)):
            use = usable[i]
            if use not in [0, 1]:
                continue

            if not force:
                if input_format == "excel":
                    x1 = self.process_traj(data['x1'][i])
                    y1 = self.process_traj(data['y1'][i])
                    ori1 = self.process_traj(data['ori1'][i])
                    x2 = self.process_traj(data['x2'][i])
                    y2 = self.process_traj(data['y2'][i])
                    ori2 = self.process_traj(data['ori2'][i])
                elif input_format == "pkl":
                    x1 = data['x1'][i]
                    y1 = data['y1'][i]
                    ori1 = data['ori1'][i]
                    x2 = data['x2'][i]
                    y2 = data['y2'][i]
                    ori2 = data['ori2'][i]
                if self.reverse: #YY
                    x1.reverse()
                    y1.reverse()
                    ori1.reverse()
                    x2.reverse()
                    y2.reverse()
                    ori2.reverse()

                vx1 = []
                vy1 = []
                vx2 = []
                vy2 = []
                for j in range(len(x1)):
                    if j == len(x1) - 1:
                        vx1.append(vx1[-1])
                        vy1.append(vy1[-1])
                        vx2.append(vx2[-1])
                        vy2.append(vy2[-1])
                    else:
                        v1 = [x1[j+1] - x1[j], y1[j+1] - y1[j]]
                        v2 = [x2[j+1] - x2[j], y2[j+1] - y2[j]]
                        vx1.append(v1[0] * self.fps)
                        vy1.append(v1[1] * self.fps)
                        vx2.append(v2[0] * self.fps)
                        vy2.append(v2[1] * self.fps)

                d = np.array([x1, y1, ori1, vx1, vy1, x2, y2, ori2, vx2, vy2])
            else:
                if input_format == "excel":
                    epsilon = self.process_traj(data['epsilon'][i])
                    sigma = self.process_traj(data['sigma'][i])
                    bcoef = self.process_traj(data['bcoef'][i])
                elif input_format == "pkl":
                    epsilon_selfA = data['epsilon_selfA'][i]
                    sigma_selfA = data['sigma_selfA'][i]
                    bcoef_selfA = data['bcoef_selfA'][i]
                    epsilon_inter = data['epsilon_inter'][i]
                    sigma_inter = data['sigma_inter'][i]
                    bcoef_inter = data['bcoef_inter'][i]
                    epsilon_selfB = data['epsilon_selfB'][i]
                    sigma_selfB = data['sigma_selfB'][i]
                    bcoef_selfB = data['bcoef_selfB'][i]
                d = np.array([epsilon_selfA, sigma_selfA, bcoef_selfA, epsilon_inter, sigma_inter, bcoef_inter, epsilon_selfB, sigma_selfB, bcoef_selfB])

            d = np.transpose(d)
            
            id = int(data['id'][i])
            label = data['label'][i]
            if not force:
                # select rows in d so that it takes step of 5, starting from the 6th frame, end at the n-5th frame
                if len(d) > 16:
                    d_selected = d[5:len(d)-5-1:5]
                else:
                    d_selected = d[int(np.floor(len(d)/2))-1].reshape(1, 10)
                cur_data = {'data': d_selected, 'id': id, 'label': label}
            else:
                cur_data = {'data': d, 'id': id, 'label': label}
            self.data.append(cur_data)

            self.id_label_mapping[id] = label
        if norm:
            self.normalize()

        new_data = []
        if test_file is not None:
            if not infering:
                val_videos = pd.read_excel(test_file, engine="openpyxl")
                self.test_video_id = list(val_videos['ID'])
                
                if training:
                    for d in self.data:
                        id = d['id']
                        if id not in self.test_video_id:
                            # if d['data'].shape[0] > 1:
                            new_data.append(d)
                else:
                    infer_videos = pd.read_excel(infer_file, engine="openpyxl")
                    self.infer_video_id = list(infer_videos['ID'])
                    for id in self.test_video_id:
                        for d in self.data:
                            if d['id'] == id and id not in self.infer_video_id:
                                # if d['data'].shape[0] > 1:
                                new_data.append(d)
                                break
        else: #infering
            if infering:
                infer_videos = pd.read_excel(infer_file, engine="openpyxl")
                self.infer_video_id = list(infer_videos['ID'])
                for id in self.infer_video_id:
                    for d in self.data:
                        if d['id'] == id:
                            new_data.append(d)
                            break
        
        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]['id']
        data = self.data[idx]['data']
        label = self.data[idx]['label']

        if self.use_relative:
            split = int(data.shape[1] / 2)
            data = self.calculate_relative_data(data)
            data = data[:, split:]

        return data, id, label

    def calculate_relative_data(self, data):
        split = int(data.shape[1] / 2)
        first_half = data[:, :split]
        second_half = data[:, split:]
        second_half = second_half - first_half

        # change angle to the same scale
        scale = 1. if self.norm else 2 * np.pi
        ori2 = second_half[:, 2]
        ori2[ori2 < 0] += scale
        second_half[:, 2] = ori2

        data = np.concatenate([first_half, second_half], axis=1)
        return data

    def process_traj(self, traj):
        if isinstance(traj, str):
            # Safely evaluates string representations like "[1.2, 3.4]" into lists
            return ast.literal_eval(traj)
        return traj

    def normalize(self):
        data = [d['data'] for d in self.data]
        concat_data = np.concatenate(data, axis=0)
        split = int(concat_data.shape[1] / 2)
        first_half = concat_data[:, :split]
        second_half = concat_data[:, split:]
        concat_data = np.concatenate([first_half, second_half], axis=0)

        self.mean = np.mean(concat_data, axis=0, keepdims=True)
        self.sd = np.std(concat_data, axis=0, keepdims=True)
        self.min = np.min(concat_data, axis=0, keepdims=True)
        self.max = np.max(concat_data, axis=0, keepdims=True)
        for idx in range(len(self.data)):
            # self.data[idx]['data'] -= self.mean
            self.data[idx]['data'][:, :2] -= np.array([[2000, 1500]])
            self.data[idx]['data'][:, :2] /= self.sd[:, :2]
            self.data[idx]['data'][:, split:split+2] -= np.array([[2000, 1500]])
            self.data[idx]['data'][:, split:split+2] /= self.sd[:, :2]
            self.data[idx]['data'][:, 2] /= 2 * np.pi
            self.data[idx]['data'][:, split+2] /= 2 * np.pi
            self.data[idx]['data'][:, 3:5] /= self.sd[:, 3:5]
            self.data[idx]['data'][:, split+3:split+5] /= self.sd[:, 3:5]


class TripletCharade(Dataset):
    def __init__(self, filename, force=False, norm=True, augment=True, test_file=None, training=True, infering=False, use_relative=False, reverse=False, infer_file=None, input_format=None): #YY
        self.dataset = Charade(filename, force, norm, test_file, training, infering, use_relative, reverse, infer_file, input_format) #YY
        self.augment = augment
        self.use_relative = use_relative
        self.label_to_idx = {}
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            if label not in self.label_to_idx:
                self.label_to_idx[label] = []
            self.label_to_idx[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _, label = self.dataset[idx]
        # split = int(data.shape[1] / 2)
        positive_idx = np.random.choice(self.label_to_idx[label])
        positive_data, _, positive_label = self.dataset[positive_idx]
        assert positive_label == label
        negative_class = self.random_choice_with_exclusion(list(self.label_to_idx.keys()), label)
        negative_idx = np.random.choice(self.label_to_idx[negative_class])
        negative_data, _, negative_label = self.dataset[negative_idx]
        assert negative_label != label

        return data, positive_data, negative_data

    def random_choice_with_exclusion(self, l, ex):
        lst = copy.deepcopy(l)
        lst.remove(ex)
        return np.random.choice(lst)


def process_aggregated_labels(filename):
    df = pd.read_csv(filename, sep="\t")
    df.rename(columns={df.columns[0]: "id_label"}, inplace=True)
    categories = df.columns[1:].tolist()

    result = {}
    for idx, row in df.iterrows():
        video_name = row["id_label"]
        values = row.drop("id_label").to_list()
        result[video_name] = np.array(values)
    return result, categories


class CharadeV2(Dataset):
    def __init__(self, traj_filename, video_names, label_distribution_file=None, input_mode="", norm=True,
                 use_relative=False, reverse=False, input_format = "excel"):  # YY
        self.traj_filename = traj_filename
        self.fps = 50
        self.norm = norm
        self.use_relative = use_relative
        self.reverse = reverse  # YY
        self.input_format = input_format
        self.input_mode = input_mode
        self.id_label_mapping, self.categories = None, None

        if label_distribution_file is not None:
            self.id_label_mapping, self.categories = process_aggregated_labels(label_distribution_file)

        if isinstance(video_names, str):
            video_names = read_and_concat(video_names)
        all_video_names = read_and_concat(traj_filename)

        if input_format == "excel":
            data = pd.read_excel(traj_filename, engine="openpyxl")
        elif input_format == "pkl":
            data = pd.read_pickle(traj_filename)
        usable = data['usable']

        self.data = []
        for i in range(len(usable)):
            use = usable[i]
            if use not in [0, 1] or all_video_names[i] not in video_names:
                continue

            if self.id_label_mapping is not None and all_video_names[i] not in self.id_label_mapping:
                continue

            if input_mode in ["traj", "core", "visual", "noOri", "distNoOri", "noOriNoDS", "distVel", "locOnly", "velOnly", "distOnly", "accOnly", "scrambled", "random"]:
                d = self.get_traj(data, i) 
            elif input_mode == "allForce" or input_mode == "interactiveOnly" or input_mode == "interSelfA":
                d = self.get_force(data, i)
            elif input_mode in["trajAndForce", "coreAndForce", "trajLocationOnlyAndForce", "trajNoOriAndForce"]:
                d_traj = self.get_traj(data, i)
                d_force = self.get_force(data, i)
                assert d_traj.shape[0] == d_force.shape[0], f"{int(data['id'][i])} traj length {d_traj.shape[0]} and force length {d_force.shape[0]} don't match"
                d = np.concatenate([d_traj, d_force], axis=1)
        
            id = int(data['id'][i])
            videoLabel = data['label'][i]
            label = data['label'][i] if self.id_label_mapping is None else self.id_label_mapping[all_video_names[i]]
            cur_data = {'data': d, 'id': id, 'videoLabel': videoLabel, 'label': label}
            self.data.append(cur_data)

        print(f"Processed {self.__len__()} data")
        if self.norm:
            print("Using normalized data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]['id']
        data = self.data[idx]['data']
        videoLabel = self.data[idx]['videoLabel']
        label = self.data[idx]['label']

        if self.use_relative:
            split = int(data.shape[1] / 2)
            data = self.calculate_relative_data(data)
            data = data[:, split:]

        return torch.FloatTensor(data), id, videoLabel, label

    def get_traj(self, data, i):
        if self.norm:
            keys = ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'dist_norm', 'vx1_norm', 'vy1_norm', 'vx2_norm', 'vy2_norm', 'ax1_norm', 'ay1_norm', 'ax2_norm', 'ay2_norm', 'ori1_norm', 'ori2_norm']
            x1, y1, x2, y2, dist, vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2, ori1, ori2 = [data[k][i] for k in keys]
        else:
            keys = ['x1', 'y1', 'ori1', 'x2', 'y2', 'ori2']
            if self.input_format == "excel":
                x1, y1, ori1, x2, y2, ori2 = [self.process_traj(data[k][i]) for k in keys]
            elif self.input_format == "pkl":
                x1, y1, ori1, x2, y2, ori2 = [data[k][i] for k in keys]
            dist = [math.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2) for i in range(len(x1))]

            x1, y1 = np.array(x1), np.array(y1)
            x2, y2 = np.array(x2), np.array(y2)

            def get_velocity_copy_first(pos, fps):
                # Calculate difference between adjacent elements
                diffs = np.diff(pos) * fps
                # Prepend the first calculated difference to the start of the array
                return np.insert(diffs, 0, diffs[0]) if diffs.size > 0 else diffs

            # Calculate Velocities
            vx1, vy1 = get_velocity_copy_first(x1, self.fps), get_velocity_copy_first(y1, self.fps)
            vx2, vy2 = get_velocity_copy_first(x2, self.fps), get_velocity_copy_first(y2, self.fps)

            # Calculate Accelerations (using the same logic on the velocities)
            ax1, ay1 = get_velocity_copy_first(vx1, self.fps), get_velocity_copy_first(vy1, self.fps)
            ax2, ay2 = get_velocity_copy_first(vx2, self.fps), get_velocity_copy_first(vy2, self.fps)

        if self.reverse:
            for lst in [x1, y1, ori1, vx1, vy1, x2, y2, ori2, vx2, vy2]:
                lst.reverse()

        if self.input_mode == "traj" or self.input_mode == "trajAndForce":
            d = np.array([x1, y1, ori1, vx1, vy1, x2, y2, ori2, vx2, vy2])
        elif "core" in self.input_mode:
            d = np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2, dist])
        elif self.input_mode == "visual":
            d = np.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])
        elif self.input_mode in ["noOri", "trajNoOriAndForce", "noOriNoDS"]:
            d = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])
        elif self.input_mode == "distNoOri":
            d = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2, dist])
        elif self.input_mode == "distVel":
            d = np.array([vx1, vy1, vx2, vy2, dist])
        elif self.input_mode == "trajLocationOnlyAndForce" or self.input_mode == "locOnly":
            d = np.array([x1, y1, x2, y2])
        elif self.input_mode == "velOnly":
            d = np.array([vx1, vy1, vx2, vy2])
        elif self.input_mode == "distOnly":
            d = np.array([dist])
        elif self.input_mode == "accOnly":
            d = np.array([ax1, ay1, ax2, ay2])
        elif self.input_mode == "scrambled":
            d = np.array([x1, y1, vx2, vy2])
        elif self.input_mode == "random":
            d = np.array([x1, y1, x2, y2])
            # d = np.random.permutation(d) # this only shuffled the order of four lists
            d = d[:, np.random.permutation(d.shape[1])] # this shuffles values within each of the four lists 
        d = np.transpose(d)    
        
        if self.input_mode in ["core", "coreAndForce", "visual", "noOriNoDS", "locOnly", "velOnly", "distOnly", "accOnly", "scrambled", "random"]:
            return d
        else: # select rows in d so that it takes step of 5, starting from the 6th frame, end at the n-5th frame
            intv = 5
            if len(d) < 2*intv + 1:
                intv = 2
            d_selected = d[np.arange(intv, len(d) - intv,intv)]
            return d_selected

    def get_force(self, data, i):
        keys = [
            'epsilon_selfA', 'sigma_selfA', 'bcoef_selfA',
            'epsilon_inter', 'sigma_inter', 'bcoef_inter',
            'epsilon_selfB', 'sigma_selfB', 'bcoef_selfB'
        ]

        if self.input_format == "excel":
            (epsilon_selfA, sigma_selfA, bcoef_selfA,
             epsilon_inter, sigma_inter, bcoef_inter,
             epsilon_selfB, sigma_selfB, bcoef_selfB) = [self.process_traj(data[k][i]) for k in keys]
        elif self.input_format == "pkl":
            if self.norm:
                keys = [k + '_log_norm' for k in keys]
            (epsilon_selfA, sigma_selfA, bcoef_selfA,
             epsilon_inter, sigma_inter, bcoef_inter,
             epsilon_selfB, sigma_selfB, bcoef_selfB) = [data[k][i] for k in keys]

        if self.input_mode == "interSelfA":
            d = np.array([epsilon_selfA, sigma_selfA, bcoef_selfA, epsilon_inter, sigma_inter, bcoef_inter])
        elif self.input_mode == "interactiveOnly":
            d = np.array([epsilon_inter, sigma_inter, bcoef_inter])
        elif self.input_mode in ["allForce", "coreAndForce", "trajAndForce", "trajLocationOnlyAndForce", "trajNoOriAndForce"]:
            d = np.array([epsilon_selfA, sigma_selfA, bcoef_selfA, epsilon_inter, sigma_inter, bcoef_inter, epsilon_selfB, sigma_selfB, bcoef_selfB])
        
        d = np.transpose(d)
        return d
    
    def calculate_relative_data(self, data):
        split = int(data.shape[1] / 2)
        first_half = data[:, :split]
        second_half = data[:, split:]
        second_half = second_half - first_half

        # change angle to the same scale
        scale = 1. # if self.norm else 2 * np.pi
        ori2 = second_half[:, 2]
        ori2[ori2 < 0] += scale
        second_half[:, 2] = ori2

        data = np.concatenate([first_half, second_half], axis=1)
        return data

    def process_traj(self, traj):
        if isinstance(traj, str):
            # Safely evaluates string representations like "[1.2, 3.4]" into lists
            return np.array(ast.literal_eval(traj), dtype=float)
        return traj


def cls_collate_fn(batch):
    # batch: list of (data, id, videoLabel, label)
    datas, ids, videoLabels, labels = zip(*batch)

    # convert to tensors
    datas = [torch.as_tensor(d).float() for d in datas]     # each [T, 9]
    lengths = torch.tensor([d.shape[0] for d in datas], dtype=torch.long)

    # pad to [B, T_max, 9]
    datas_padded = pad_sequence(datas, batch_first=True)    # pads with 0

    labels = torch.stack([torch.as_tensor(l).float() for l in labels])  # [B, C]

    return datas_padded, lengths, ids, videoLabels, labels


class BalancedBatchSampler(Sampler):
    """
    Yields batches of indices with class-coverage constraints:

    - If batch_size < num_classes:
        each item in a batch comes from a different class (no repeats in-batch).
    - If batch_size >= num_classes:
        each (non-empty) class appears at least once per batch, and the remaining
        slots are filled round-robin from classes that still have samples.

    Uses every sample exactly once per epoch (unless drop_last=True).
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.labels = []
        for i in range(len(dataset)):
            _, _, videoLabel, _ = dataset[i]
            self.labels.append(videoLabel)

        # Build indices per class
        class_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            class_indices[y].append(idx)

        self.class_indices = class_indices
        self.classes = sorted(self.class_indices.keys())
        self.num_classes = len(self.classes)

        self.dataset_size = len(self.labels)

        if drop_last:
            self._num_batches = self.dataset_size // self.batch_size
        else:
            self._num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        # Make per-class deques for fast pops
        pools = {}
        for c in self.classes:
            idxs = list(self.class_indices[c])
            if self.shuffle:
                np.random.shuffle(idxs)
            pools[c] = deque(idxs)

        # Active classes (still have samples)
        active = [c for c in self.classes if len(pools[c]) > 0]
        if not active:
            return

        rr_ptr = 0  # round-robin pointer over active classes

        def advance_to_next_nonempty(start_ptr):
            """Advance rr_ptr to a class that still has samples."""
            if not active:
                return 0
            p = start_ptr % len(active)
            # active list only holds non-empty pools, so this is safe
            return p

        def pop_one(c):
            """Pop one index from class c, updating active list if it becomes empty."""
            nonlocal rr_ptr
            idx = pools[c].popleft()
            if len(pools[c]) == 0:
                # remove from active and fix rr_ptr if needed
                remove_pos = active.index(c)
                active.pop(remove_pos)
                if active:
                    if remove_pos < rr_ptr:
                        rr_ptr -= 1
                    rr_ptr %= len(active)
                else:
                    rr_ptr = 0
            return idx

        while active:
            batch = []

            if self.batch_size < self.num_classes:
                # Need batch_size distinct classes. Take next distinct classes from round-robin.
                k = min(self.batch_size, len(active))
                used_classes = []

                rr_ptr = advance_to_next_nonempty(rr_ptr)
                # collect k distinct classes starting from rr_ptr
                for t in range(k):
                    c = active[(rr_ptr + t) % len(active)]
                    used_classes.append(c)

                # advance rr_ptr for next batch
                rr_ptr = (rr_ptr + k) % len(active)

                for c in used_classes:
                    batch.append(pop_one(c))

            else:
                # batch_size >= num_classes:
                # ensure each non-empty class appears at least once
                current_active_snapshot = list(active)  # stable snapshot for "at least once"
                for c in current_active_snapshot:
                    batch.append(pop_one(c))
                    if not active:  # everything exhausted exactly at boundary
                        break

                # fill remaining slots round-robin from remaining active classes
                while active and len(batch) < self.batch_size:
                    rr_ptr = advance_to_next_nonempty(rr_ptr)
                    c = active[rr_ptr]
                    batch.append(pop_one(c))
                    if active:
                        rr_ptr %= len(active)

            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch


def split_train_val(csv_file, val_per_class=6, label_col="label"):
    df = pd.read_csv(csv_file)

    # sample val_per_class rows per label (keep original indices!)
    val_df = df.groupby(label_col, group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), val_per_class))
    )

    # drop using the original indices
    train_df = df.drop(index=val_df.index)

    return read_and_concat(train_df), read_and_concat(val_df)



if __name__ == '__main__':
    one_hot = False
    excel_name = 'data/charade_traj_force_summary_full.pkl'
    all_video_file = 'data/charades_traj_all_without_inference.csv'

    train_videos, val_videos = split_train_val(all_video_file)

    print("train:", len(train_videos), "val:", len(val_videos))

    overlap = set(train_videos) & set(val_videos)

    if overlap:
        raise ValueError(f"Overlap found ({len(overlap)}): {sorted(overlap)[:20]}")
    else:
        print("No overlap.")
