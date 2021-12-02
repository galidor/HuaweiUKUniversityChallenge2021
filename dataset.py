from torch.utils.data import Dataset
import os.path as osp
import json
import pandas as pd
import numpy as np


class HuaweiDataset(Dataset):
    def __init__(self, data_path='data', split='train'):
        self.data_path = data_path
        self.split = split # 'train', 'val', 'trainval'
        # Load fingerprints
        with open(osp.join(self.data_path, "task1_data", "task1_fingerprints.json")) as f:
            self.fingerprints = json.load(f)
        if self.split in ['train', 'val', 'trainval']:
            # Load train pairs
            self.idx_pairs = pd.read_csv(osp.join(self.data_path, "task1_data", "task1_train.csv"))
            if self.split in ['train', 'trainval']:
                # Remove data points that are far away from each other (>100m)
                self.idx_pairs = self.idx_pairs.loc[self.idx_pairs.loc[:, 'displacement'] < 100.]
            # Train/val split
            np.random.seed(2136) # Random seed ensures unchanging validation set
            random_indices = np.random.permutation(self.idx_pairs.index.values)
            if self.split == 'train':
                self.indices = random_indices[:-10000]
            elif self.split == 'val':
                self.indices = random_indices[-10000:]
            elif self.split == 'trainval':
                self.indices = random_indices
        elif self.split == 'test':
            # Loading test data
            self.idx_pairs = pd.read_csv(osp.join(self.data_path, "task1_data", "task1_test.csv"))
            self.indices = np.arange(self.idx_pairs.shape[0])
        else:
            raise NotImplementedError

        # Normalize the RSSI by finding the lowest value and dividing by it (division is performed in __getitem__())
        max_rssi = -100.
        min_rssi = 0.
        for key in self.fingerprints:
            for mac_address in self.fingerprints[key]:
                if self.fingerprints[key][mac_address] < min_rssi:
                    min_rssi = self.fingerprints[key][mac_address]
                if self.fingerprints[key][mac_address] > max_rssi:
                    max_rssi = self.fingerprints[key][mac_address]

        self.min_rssi = min_rssi
        self.max_rssi = max_rssi

        # Read named MAC addresses. We performed search over MAC addresses in the dataset and we keep only those, which we claim to be WiFi routers
        with open('invalid_mac_addresses.json') as f:
            invalid_mac_addresses = json.load(f)
        invalid_mac_addresses = [int(invalid_mac_address.replace(':', ''), 16) for invalid_mac_address in invalid_mac_addresses]

        # Only consider the most common mac addresses (static ones)
        mac_addresses_count = {}
        for key in self.fingerprints:
            for mac_address in self.fingerprints[key]:
                if mac_address in mac_addresses_count:
                    mac_addresses_count[mac_address] += 1
                else:
                    mac_addresses_count[mac_address] = 1

        self.mac_addresses = [mac_address for mac_address in mac_addresses_count if
                              mac_addresses_count[mac_address] > 10]

        self.mac_addresses = [mac_address for mac_address in self.mac_addresses if mac_address not in invalid_mac_addresses]

    def export_results(self, preds):
        # Saves predictions into .csv format
        submission_df = self.idx_pairs
        submission_df['displacement'] = preds
        submission_df.to_csv('my_submission.csv', index=False)

    def __getitem__(self, idx):
        '''
        Our dataloader provides up to 10 common MAC addresses for both fingerprints, if they exist. Otherwise it starts considering
        disjoint sets of MAC addresses by filling the power of the WiFi AP, if available and -150 dB in the field corresponding to
        the second fingerprint.
        '''
        idx = self.indices[idx]
        # Assume maximum number of 10 common MAC addresses
        common_signal_powers = []
        mac_similarity = 0 # Specifies the number of common MAC addresses
        id1_hotspots = self.fingerprints[str(self.idx_pairs.loc[idx, 'id1'])]
        id2_hotspots = self.fingerprints[str(self.idx_pairs.loc[idx, 'id2'])]
        for hotspot in id1_hotspots:
            if hotspot in id2_hotspots:
                common_signal_powers.append([id1_hotspots[hotspot], id2_hotspots[hotspot]])
                # If there exist a common MAC address with small transmit power, only add 0.5 to mac_similarity
                if id1_hotspots[hotspot] > -55 or id2_hotspots[hotspot] > -55:
                    mac_similarity += 1.
                else:
                    mac_similarity += 0.5
            else:
                common_signal_powers.append([id1_hotspots[hotspot], -150.])
        for hotspot in id2_hotspots:
            if hotspot not in id1_hotspots:
                common_signal_powers.append([-150., id2_hotspots[hotspot]])

        # Repeat some values if we don't have enough measurements for a given fingerprint pair
        threshold = 10
        if len(common_signal_powers) < threshold:
            for i in range(threshold - len(common_signal_powers)):
                common_signal_powers.append(common_signal_powers[i])
        elif len(common_signal_powers) > threshold:
            common_signal_powers = common_signal_powers[:threshold]

        # Return absolute difference bettwen fingerprint indices, mac_similarity and the powers of the received WiFi signals
        if self.split in ['train', 'val']:
            return np.concatenate([np.array([self.idx_pairs.loc[idx, 'id1'] - self.idx_pairs.loc[idx, 'id2']], dtype=np.float32), np.array([mac_similarity], dtype=np.float32),
                                   np.array(common_signal_powers, dtype=np.float32).reshape((20)) / self.min_rssi]), self.idx_pairs.loc[idx, 'displacement'].astype(np.float32)

        elif self.split == 'test':
            return np.concatenate([np.array([self.idx_pairs.loc[idx, 'id1'] - self.idx_pairs.loc[idx, 'id2']], dtype=np.float32), np.array([mac_similarity], dtype=np.float32),
                        np.array(common_signal_powers, dtype=np.float32).reshape((20)) / self.min_rssi]), np.array(0., dtype=np.float32)

    def __len__(self):
        return self.indices.shape[0]
