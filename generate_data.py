import numpy as np
import pathlib
import sys
import warnings
from functools import partial as bind
from collections import namedtuple

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

from dreamerv3 import embodied
from dreamerv3.train import make_logger, make_envs
from dreamerv3.recoder import RecordMP4JSONEnv

# model_path = "logdir/dmc_walker_walk"
# dataset_dir = "/data/expertdata/train/dmc_walker_walk"

model_path = "logdir/atari_pong"
dataset_dir = "/data/expertdata/train/atari_pong"


def eval_only(agent, env, args):

    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        driver(policy, steps=100)


if __name__ == "__main__":
    from dreamerv3 import agent as agt

    config = embodied.Config.load(model_path + "/config.yaml")
    config = config.update({"envs.amount": 1})
    config = config.update({"jax.policy_devices": (2,), "jax.train_devices": (2,)})
    print(config)

    step = embodied.Counter()

    # update_config = embodied.Config({'env': {'crafter': {'outdir': "recorded"}}})

    # config.update(update_config)

    cleanup = []
    env_native = make_envs(config)  # mode='eval'
    env = RecordMP4JSONEnv(env_native, dataset_dir, parallel=(config.envs.parallel != "none"))
    cleanup.append(env)
    agent = agt.Agent(env.obs_space, env.act_space, step, config)
    args = embodied.Config(
        logdir=config.logdir,
        from_checkpoint=model_path + "/checkpoint.ckpt",
        steps=10000000,
    )
    eval_only(agent, env, args)

    for obj in cleanup:
        obj.close()
