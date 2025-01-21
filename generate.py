import numpy as np
import pathlib
import sys
import warnings
import argparse
from functools import partial as bind
from collections import namedtuple

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

from dreamerv3 import embodied
from dreamerv3.train import make_logger, make_envs
from dreamerv3.recoder import RecordMP4JSONEnv


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
    # Argument parser for model_path and dataset_dir
    parser = argparse.ArgumentParser(description="Evaluation script for DreamerV3 model.")
    parser.add_argument("--model", required=True, help="Path to the model directory.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory.")
    args = parser.parse_args()

    from dreamerv3 import agent as agt

    config = embodied.Config.load(args.model + "/config.yaml")
    config = config.update({"envs.amount": 1})
    config = config.update({"jax.policy_devices": (0,), "jax.train_devices": (0,)})
    print(config)

    step = embodied.Counter()

    cleanup = []
    env_native = make_envs(config)  # mode='eval'
    env = RecordMP4JSONEnv(env_native, args.dataset, parallel=(config.envs.parallel != "none"))
    cleanup.append(env)
    agent = agt.Agent(env.obs_space, env.act_space, step, config)
    eval_args = embodied.Config(
        logdir=config.logdir,
        from_checkpoint=args.model + "/checkpoint.ckpt",
        steps=10000000,
    )
    eval_only(agent, env, eval_args)

    for obj in cleanup:
        obj.close()

# python generate_data.py --model_path logdir/atari_pong --dataset_dir /data/expertdata/train/atari_pong
