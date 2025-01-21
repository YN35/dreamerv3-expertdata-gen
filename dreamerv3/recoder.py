import os
import gzip
import json
import uuid
import numpy as np
import imageio

import embodied


class RecordMP4JSONEnv(embodied.Env):
    def __init__(self, env, output_dir, parallel, save_interval=30, recode_keys=[]):
        self.env = env
        self.output_dir = output_dir
        self.parallel = parallel
        self.save_interval = save_interval
        self.episode_buffers = []
        self.current_episode = 0
        self.recode_keys = ["reward", "action", "image"]
        self.recode_keys.extend(recode_keys)
        os.makedirs(self.output_dir, exist_ok=True)

        self.discrete_action = False
        if all(self.act_space["action"].high == 1) and all(self.act_space["action"].low == 0):
            self.discrete_action = True

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
        if action["reset"].any():
            if len(self.episode_buffers) >= self.save_interval:
                self._save_episodes_data()
                self.episode_buffers.clear()

            self.current_episode += 1
            self._start_new_episode()
        obs = self.env.step(action)
        self._record_obs(obs, action)

        return obs

    def _record_obs(self, obs, action):
        recode = {}
        obs_copy = obs.copy()
        obs_copy["action"] = action["action"]
        if self.discrete_action:
            obs_copy["action"] = np.argmax(obs_copy["action"], axis=-1)
        for key in self.recode_keys:
            recode[key] = obs_copy[key]
        self.episode_buffers[-1].append(recode)

    def _start_new_episode(self):
        self.episode_buffers.append([])

    def _save_episodes_data(self):
        num_episodes_to_save = len(self.episode_buffers)
        batch_size = len(self.episode_buffers[0][0]["action"])

        for e in range(num_episodes_to_save):
            for b in range(batch_size):
                episode_uid = str(uuid.uuid4())
                video_filename = os.path.join(self.output_dir, f"{episode_uid}.mp4")

                json_data = {}
                for key in self.episode_buffers[e][0]:
                    data = np.array([obs_action_pair[key][b] for obs_action_pair in self.episode_buffers[e]])
                    if key == "image":
                        with imageio.get_writer(video_filename, fps=10) as writer:
                            for obs_action_pair in self.episode_buffers[e]:
                                writer.append_data(obs_action_pair[key][b])
                    else:
                        data_list = data.tolist()
                        json_data[key] = data_list

                episode_length = len(self.episode_buffers[e])
                json_data["episode_length"] = episode_length
                json_data["reward_sum"] = np.sum(json_data["reward"])

                # JSON data for other data
                json_filename = os.path.join(self.output_dir, f"{episode_uid}.json.gz")
                with gzip.open(json_filename, mode="wt", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

    def render(self):
        return self.env.render()

    def close(self):
        if self.episode_buffers:
            self._save_episodes_data()
        self.env.close()
