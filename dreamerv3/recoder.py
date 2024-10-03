import os
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

                    # JSON data for other data
                    json_filename = os.path.join(self.output_dir, f"{episode_uid}.json")
                    with open(json_filename, "w") as json_file:
                        json.dump(json_data, json_file)

    def render(self):
        return self.env.render()

    def close(self):
        if self.episode_buffers:
            self._save_episodes_data()
        self.env.close()
