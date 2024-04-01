import importlib
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
from dreamerv3.embodied import wrappers
from dreamerv3.train import make_logger, make_replay, make_envs
from dreamerv3.recoder import RecordHDF5Env, RecordMP4JSONEnv

model_path = "models/crafter0"
logdir = "logs/crafter0-1"
dataset_dir = "/home/ynn/datasets/crafter_expertdata/val"
# dataset_dir = "test/"

if __name__ == "__main__":
    from dreamerv3 import agent as agt

    config = embodied.Config.load(model_path + "/config.yaml")
    print(config)

    logdir = embodied.Path(logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    step = embodied.Counter()
    logger = make_logger(None, logdir, step, config)

    # update_config = embodied.Config({'env': {'crafter': {'outdir': "recorded"}}})

    # config.update(update_config)

    cleanup = []
    env_native = make_envs(config)  # mode='eval'
    env = RecordMP4JSONEnv(env_native, dataset_dir)
    cleanup.append(env)
    agent = agt.Agent(env.obs_space, env.act_space, step, config)
    args = embodied.Config(
        logdir=config.logdir,
        log_every=30,  # Seconds
        log_keys_video=["image"],
        log_zeros=False,
        log_keys_sum=".*",
        log_keys_mean=".*",
        log_keys_max=".*",
        from_checkpoint=model_path + "/checkpoint.ckpt",
        steps=1000000000000,
    )
    embodied.run.eval_only(agent, env, logger, args)

    for obj in cleanup:
        obj.close()
