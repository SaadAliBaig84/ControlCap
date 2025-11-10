import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# --- ControlCap registrations (keep order; these import and register components) ---
from controlcap.tasks import *        # noqa: F401,F403
from controlcap.datasets import *     # noqa: F401,F403
from controlcap.models import *       # noqa: F401,F403
from controlcap.runners import *      # noqa: F401,F403
from controlcap.common.config import Config

# --- LAVIS infra ---
import lavis.tasks as tasks
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# (explicit imports so registry side-effects happen)
from lavis.datasets.builders import *   # noqa: F401,F403
from lavis.models import *              # noqa: F401,F403
from lavis.processors import *          # noqa: F401,F403
from lavis.runners import *             # noqa: F401,F403
from lavis.tasks import *               # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description="ControlCap Train/Eval")

    parser.add_argument("--cfg-path", required=True, help="Path to YAML config.")
    parser.add_argument("--local-rank", default=-1, type=int, help="For torch.distributed (debug ok).")

    # Override any config field on the CLI:
    # Example:
    #   --options run.batch_size_eval=1 model.tag_chunk_size=4 model.num_beams=1
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "Override settings in the config. Use key=value pairs; nested with dots. "
            "Examples: run.batch_size_eval=1 model.tag_chunk_size=4"
        ),
    )

    return parser.parse_args()


def setup_seeds(config):
    """Make runs reproducible across ranks."""
    seed = int(config.run_cfg.seed) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """Runner choice is controlled by run.task; default to 'controlcap'."""
    return registry.get_runner_class(cfg.run_cfg.get("task", "controlcap"))


def main():
    # Single job id shared across ranks (set before init_distributed_mode).
    job_id = now()

    # Parse CLI and build config
    args = parse_args()
    cfg = Config(args)

    # Init distributed (or noop if world size == 1)
    init_distributed_mode(cfg.run_cfg)

    # Seeds after dist init
    setup_seeds(cfg)

    # Logger only talks on master
    setup_logger()

    # Print effective config once
    cfg.pretty_print()

    # Build task/datasets/model via registry
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # Build runner and go
    runner_cls = get_runner_class(cfg)
    runner = runner_cls(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    main()
