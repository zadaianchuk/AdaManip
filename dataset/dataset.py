import numpy as np
import torch
import pickle
from pytorch3d.ops import sample_farthest_points
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import zarr
import ipdb

def obs_wrapper(obs, dof=0):
    pcs = obs['pc']
    device = pcs.device
    if dof == 0:
        dof_state = torch.tensor([]).to(device)
    elif dof == 1:
        dof_state = obs['dof_state'][1:]  # do not reserve the dof state of the prismatic joint
    elif dof == 2:
        dof_state = obs['dof_state']
    env_state = torch.cat([obs['proprioception'], dof_state, obs['prev_action']], dim=-1)
    return pcs, env_state

class Episode_Buffer:
    def __init__(self):
        self.pcs = []
        self.env_state = []
        self.action = []

    def add(self, pc, env_state, action):
        #pc, env_state = obs_wrapper(obs)
        self.pcs.append(pc.cpu().numpy())
        self.env_state.append(env_state.cpu().numpy())
        self.action.append(action.cpu().numpy())

def nested_dict_save(zarr_group, data, path):
    for key,value in data.items():
        if isinstance(value, dict):
            nested_dict_save(zarr_group, value, path+'/'+key)
        else:
            zarr_group[path+'/'+key] = value

class Experience:
    def __init__(self, sample_pcs_num=1000):
        self.sample_pcs_num = sample_pcs_num
        self.data = {"pcs":[], "env_state": [], "action": []}
        self.meta = {"episode_ends": []}
    
    def append(self, episode:Episode_Buffer):
        # new_pcs = torch.tensor(episode.pcs)
        # downsample, idx = sample_farthest_points(new_pcs, self.sample_pcs_num)
        if len(episode.pcs) == 0:
            # print("skip empty eps")
            return        
        if self.meta["episode_ends"] == []:
            self.data["pcs"] = np.array(episode.pcs)
            self.data["env_state"] = np.array(episode.env_state)
            self.data["action"] = np.array(episode.action)
        else:
            self.data["pcs"] = np.concatenate([self.data["pcs"], np.array(episode.pcs)])
            self.data["env_state"] = np.concatenate([self.data["env_state"], np.array(episode.env_state)])
            self.data["action"] = np.concatenate([self.data["action"], np.array(episode.action)])
        # self.data["pcs"] = self.data["pcs"] + episode.pcs
        # self.data["pose"] = self.data["pose"] + episode.pose
        # self.data["action"] = self.data["action"] + episode.action
        new_end = self.data["pcs"].shape[0]
        # print(new_end)
        self.meta["episode_ends"].append(new_end)

    def save(self, path):
        # save data and meta in one file as npy
        self.meta["episode_ends"] = np.array(self.meta["episode_ends"])
        #data_save = {"data": self.data, "meta": self.meta}
        zarr_group = zarr.open(path, 'w')
        nested_dict_save(zarr_group, self.data, 'data')
        nested_dict_save(zarr_group, self.meta, 'meta')

def create_sample_indices(episode_ends:np.ndarray, obs_length:int, action_length: int, pad_after: int):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        pad_before = obs_length
        
        # action length is the pred_horizon, obs length is the obs_horizon, pad_after is the action_horizon
        for idx in range(episode_length-action_length+pad_after):
            action_start_idx = idx + start_idx
            action_end_idx = min(idx+action_length,episode_length) + start_idx
            obs_start_idx = max(idx-obs_length+1,0) + start_idx
            obs_end_idx = idx + 1 + start_idx

            indices.append([action_start_idx, action_end_idx, obs_start_idx, obs_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, obs_length, action_length,
                    action_start_idx, action_end_idx, obs_start_idx, obs_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        if key != "action":
            sample = input_arr[obs_start_idx:obs_end_idx]
            if obs_end_idx - obs_start_idx < obs_length:
                data = np.zeros(shape=(obs_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                data[:obs_length-obs_end_idx+obs_start_idx] = sample[0]
                data[obs_length-obs_end_idx+obs_start_idx:] = sample
            else:
                data = sample
        else:
            sample = input_arr[action_start_idx:action_end_idx]
            if action_end_idx - action_start_idx < action_length:
                data = np.zeros(shape=(action_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                data[:action_end_idx-action_start_idx] = sample
                data[action_end_idx-action_start_idx:] = sample[-1]
            else:
                data = sample

        # sample = input_arr[buffer_start_idx:buffer_end_idx]
        # data = sample
        # if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
        #     data = np.zeros(
        #         shape=(sequence_length,) + input_arr.shape[1:],
        #         dtype=input_arr.dtype)
        #     if sample_start_idx > 0:
        #         data[:sample_start_idx] = sample[0]
        #     if sample_end_idx < sequence_length:
        #         data[sample_end_idx:] = sample[-1]
        #     data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# dataset
class ManipDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: list,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        # save obs and action
        # obs: global point clouds, joint pose, gripper pose
        # action: joint pose, joint velocity, gripper force
        # read from npy dataset
        print("using new Manip Dataset")
        print(f"loading data from {dataset_path}")
        pcs_list = []
        action_data_list = []
        env_state_list = []
        episode_end_list = []

        init = 0
        for data_path in dataset_path:
            data = zarr.open(data_path, 'a')
            ends = data['meta']['episode_ends'][:]
            episode_end_list.append(ends + init)
            init += ends[-1]
            pcs_list.append(data['data']['pcs'])
            action_data_list.append(data['data']['action'])
            env_state_list.append(data['data']['env_state'])

        # pcs = np.moveaxis(dataset_root['data']['pcs'][:], 1, -1)
        '''
        seg pointcloud don not need moveaxis
        '''
        # ipdb.set_trace()
        pcs = np.concatenate(pcs_list, axis=0)
        action_data = np.concatenate(action_data_list, axis=0)
        env_state = np.concatenate(env_state_list, axis=0)
        # max length == 7
        # action_pos = action_data[:,:8]
        #env_state_data = np.concatenate([dataset_root['data']['env_state'][:,:25], dataset_root['data']['env_state'][:,26:]], axis=-1)
        # action_gripper = action_data[:,14:16]/10  # normalize to [-1,1]
        # print(action_gripper)
        # action_train = np.concatenate([action_pos, action_gripper], axis=1)
        action_train = action_data
        train_data = {
            'pcs': pcs,
            'env_state': env_state,
            'action': action_train#dataset_root['data']['action'][:]
        }
        episode_ends = np.concatenate(episode_end_list, axis=0)
        # ipdb.set_trace()
        # print(episode_ends)
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            obs_length= obs_horizon,
            action_length= pred_horizon,
            pad_after= pred_horizon #action_horizon-1
        )
        print("pad after is", pred_horizon)
        self.indices = indices
        # print(train_data_len)
        self.train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # compute which segmentation of data is idx in 
        # seg = 0
        # for i in range(len(self.train_data_len)):
        #     if idx < self.train_data_len[i]:
        #         seg = i
        #         break
        #     else:
        #         idx -= self.train_data_len[i]
        #print(seg)
        # get the start/end indices for this datapoint
        action_start_idx, action_end_idx, obs_start_idx, obs_end_idx = self.indices[idx]
        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            obs_length=self.obs_horizon,
            action_length=self.pred_horizon,
            action_start_idx=action_start_idx,
            action_end_idx=action_end_idx,
            obs_start_idx=obs_start_idx,
            obs_end_idx=obs_end_idx
        )

        # discard unused observations
        # nsample['pcs'] = nsample['pcs'][:self.obs_horizon,:]
        # nsample['pose'] = nsample['pose'][:self.obs_horizon,:]
        return nsample

def merge_dataset(path_list, to_path):
    for path in path_list:
        dataset_root = zarr.open(path, 'r')
        if path == path_list[0]:
            pcs_data = dataset_root['data']['pcs'][:]
            pose_data = dataset_root['data']['env_state'][:]
            action_data = dataset_root['data']['action'][:]
            meta = dataset_root['meta']['episode_ends'][:]
            cur_len = dataset_root['meta']['episode_ends'][-1]
        else:
            pcs_data = np.concatenate([pcs_data, dataset_root['data']['pcs'][:]])
            pose_data = np.concatenate([pose_data, dataset_root['data']['env_state'][:]])
            action_data = np.concatenate([action_data, dataset_root['data']['action'][:]])
            new_meta = dataset_root['meta']['episode_ends'][:] + cur_len
            cur_len += dataset_root['meta']['episode_ends'][-1]
            meta = np.concatenate([meta, new_meta])
            #print(new_meta)
        print(pcs_data.shape, pose_data.shape, action_data.shape, meta.shape)

    # save data and meta in one file using zarr
    all_data = {"pcs":pcs_data, "env_state": pose_data, "action": action_data}
    all_meta = {"episode_ends": meta}
    zarr_group = zarr.open(to_path, 'w')
    nested_dict_save(zarr_group, all_data, 'data')
    nested_dict_save(zarr_group, all_meta, 'meta')


if __name__ == "__main__":
    # merge dataset
    # path_prefix = "./demo_data/ada/nohis/"
    # to_path = "./demo_data/ada/nohis/demo_mix.zip"
    # obj_id = ["005", "012", "013", "022", "031", "065"]
    # # append obj_id to prefix to get each file path
    # path_list = [path_prefix + id + "/demo_buffer.zip" for id in obj_id]
    to_path = "./demo_data/succ/51213223154full.zip"
    path_list = []
    path_list.append("./demo_data/succ/512133165/demo_buffer.zip")  # another half is success
    path_list.append("./demo_data/succ/022_100/demo_buffer.zip") 
    merge_dataset(path_list,to_path)

    # test dataset
    # dataset_path = "./demo_data/ada/01-12-01-07-38/demo_buffer.zip"
    # dataset_root = zarr.open(dataset_path, 'r')
    #dataset = ManipDataset(dataset_path, 16, 2, 8)
    # eps_end = np.array(dataset_root['meta']['episode_ends'][:])
    # print(eps_end)
    #print(dataset_root['data']['pose'][0,25:])
    # points = torch.tensor(dataset_root['data']['pcs'][0]).unsqueeze(0)
    # pcs = Pointclouds(points=points)
    # fig = plot_scene({"plot1":{"pcs":pcs}})
    # fig.show()
    # episode_end = np.array([20])
    # idxs = create_sample_indices(episode_end, 2, 16, 8)
    # print(idxs)
