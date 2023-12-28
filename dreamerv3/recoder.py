import h5py
import numpy as np
import json

import embodied


class RecordHDF5Env(embodied.Env):
    def __init__(self, env, hdf5_filename, save_interval=30):
        self.env = env
        self.hdf5_filename = hdf5_filename
        self.save_interval = save_interval
        self.episode_buffers = []
        self.current_episode = 0

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

        if obs["is_first"]:
            self.current_episode += 1
            self._start_new_episode()

        self._record_obs(obs, action)

        if obs["is_last"]:
            if len(self.episode_buffers) >= self.save_interval:
                self._save_episodes_data()
                self.episode_buffers.clear()

        return obs

    def _record_obs(self, obs, action):
        obs_copy = obs.copy()
        obs_copy["action"] = action["action"]
        obs_copy["is_first"] = obs_copy["is_first"]
        self.episode_buffers[-1].append((obs_copy))

    def _start_new_episode(self):
        self.episode_buffers.append([])

    def _save_episodes_data(self):
        with h5py.File(self.hdf5_filename, "a") as file:
            num_episodes_to_save = len(self.episode_buffers)

            for i in range(num_episodes_to_save):
                episode_name = (
                    f"episode_{self.current_episode - num_episodes_to_save + i}"
                )
                episode_group = file.create_group(episode_name)

                for key in self.episode_buffers[i][0]:
                    data = np.array(
                        [
                            obs_action_pair[key][0]
                            for obs_action_pair in self.episode_buffers[i]
                        ]
                    )
                    if key == "image":
                        data = data.astype(np.uint8)
                    else:
                        data = data.astype(np.float32)
                    episode_group.create_dataset(
                        key,
                        data=data,
                        compression="gzip",
                        compression_opts=4,
                        chunks=True,
                    )
                episode_length = len(self.episode_buffers[i])
                episode_group.create_dataset(
                    "episode_length",
                    data=np.array([episode_length], dtype=np.int32),
                )

    def render(self):
        return self.env.render()

    def close(self):
        if self.episode_buffers:
            self._save_episodes_data()
        self.env.close()
