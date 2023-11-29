import h5py
import numpy as np
import json

import embodied


"""
network
  o_t-1  o_t  o_t+1
    |     |     |
    v     v     v
   a_t  a_t+1  a_t+2
env
    |     |     |
    v     v     v
   o_t  o_t+1  END

data
  o_t-1 | o_t | o_t+1
  a_t-1 | a_t | a_t+1
"""


class RecordHDF5Env(embodied.Env):
    def __init__(self, env, hdf5_filename, buffer_size=1000):
        self.env = env
        self.hdf5_filename = hdf5_filename
        self.file = h5py.File(hdf5_filename, "w")
        self.data_group = self.file.create_group("obs")
        self.buffer = []
        self.buffer_size = buffer_size
        self.frame_count = 0

    def __len__(self):
        return len(self.env)

    def __bool__(self):
        return bool(self.env)

    @property
    def obs_space(self):
        return self.env.obs_space

    @property
    def act_space(self):
        return self.env.act_space

    def step(self, action):
        obs = self.env.step(action)
        self._record_obs(obs, action)
        if obs["is_first"]:
            self._flush_buffer(is_first=True)
        elif len(self.buffer) >= self.buffer_size:
            self._flush_buffer(is_first=False)
        return obs

    def _record_obs(self, obs, action, is_first=False):
        self.buffer.append(obs)
        self.frame_count += 1

    def _flush_buffer(self, is_first=False):
        if is_first:
            if "start_idx" not in self.data_group:
                self.data_group.create_dataset(
                    "start_idx", data=[self.frame_count], maxshape=(None,)
                )

        for key in self.buffer[0]:
            data = np.array([obs[key] for obs in self.buffer])

            if key in self.data_group:
                dset = self.data_group[key]
                dset.resize((dset.shape[0] + data.shape[0],) + dset.shape[1:])
                dset[-data.shape[0] :] = data
            else:
                self.data_group.create_dataset(
                    key, data=data, maxshape=(None,) + data.shape[1:]
                )

        self.buffer = []

    def render(self):
        return self.env.render()

    def close(self):
        self._flush_buffer()
        self.file.close()
        self.env.close()

    def __del__(self):
        self.close()
