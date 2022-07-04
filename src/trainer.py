from src.utils import get_benchmark_by_name, seed_everything
import os
import time
import numpy as np
import logging


class Trainer():
    def __init__(self, args):
        self.args = args
        self._build()

    def _build(self):
        seed_everything()
        self._create_config_file()
        train_function = get_benchmark_by_name(algo_name=self.args.algo,
                                               env_name=self.args.env)
        train_function(self.args)

    def _create_config_file(self):
        if (self.args.snapshot_dir is not None):
            if not os.path.exists(self.args.snapshot_dir):
                os.makedirs(self.args.snapshot_dir)
            folder = os.path.join(self.args.snapshot_dir,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            self.args.snapshot_dir = os.path.abspath(folder)
