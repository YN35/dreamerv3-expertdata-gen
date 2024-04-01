import os
import json
import h5py
import numpy as np
import imageio

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
                        with imageio.get_writer("temp.mp4", fps=20) as writer:
                            for obs_action_pair in self.episode_buffers[i]:
                                writer.append_data(obs_action_pair[key][0])
                        with open("temp.mp4", "rb") as mp4_file:
                            mp4_data = mp4_file.read()
                            data = np.void(mp4_data)
                    else:
                        data = data.astype(np.float32)
                    episode_group.create_dataset(key, data=data)
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


class RecordMP4JSONEnv(embodied.Env):
    def __init__(self, env, output_dir, save_interval=30, recode_keys=[]):
        self.env = env
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.episode_buffers = []
        self.current_episode = 0
        self.recode_keys = ["reward", "action", "image", "is_first", "is_last"]
        self.recode_keys.extend(recode_keys)
        os.makedirs(self.output_dir, exist_ok=True)

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
        recode = {}
        obs_copy = obs.copy()
        obs_copy["action"] = action["action"]
        for key in self.recode_keys:
            recode[key] = obs_copy[key]
        self.episode_buffers[-1].append(recode)

    def _start_new_episode(self):
        self.episode_buffers.append([])

    def _save_episodes_data(self):
        num_episodes_to_save = len(self.episode_buffers)

        for i in range(num_episodes_to_save):
            episode_num = self.current_episode - num_episodes_to_save + i
            episode_name = str(episode_num).zfill(8)
            video_filename = os.path.join(self.output_dir, f"{episode_name}.mp4")

            json_data = {}
            for key in self.episode_buffers[i][0]:
                data = np.array(
                    [
                        obs_action_pair[key][0]
                        for obs_action_pair in self.episode_buffers[i]
                    ]
                )
                if key == "image":
                    with imageio.get_writer(video_filename, fps=10) as writer:
                        for obs_action_pair in self.episode_buffers[i]:
                            writer.append_data(obs_action_pair[key][0])
                else:
                    data_list = data.tolist()
                    json_data[key] = data_list

            episode_length = len(self.episode_buffers[i])
            json_data["episode_length"] = episode_length

            # JSON data for other data
            json_filename = os.path.join(self.output_dir, f"{episode_name}.json")
            with open(json_filename, "w") as json_file:
                json.dump(json_data, json_file)

    def render(self):
        return self.env.render()

    def close(self):
        if self.episode_buffers:
            self._save_episodes_data()
        self.env.close()
