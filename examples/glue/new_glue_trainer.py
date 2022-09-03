# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Specifically designed for the NLP use case, allowing pre-training and fine-tuning on
downstream tasks to be handled within one script. This script requires that the
run_composer_trainer.py script lies in the parent folder to this one.

Example that pretrains a BERT::
    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme pretrain

Example that pretrains and finetunes a BERT::
    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme all

Example that finetunes a pretrained BERT::

    >>> python examples/glue/run_glue_trainer.py
    -f examples/glue/glue_example.yaml
    --training_scheme finetune

To see all the possible options for a specific parameter usage,
try ``python examples/glue/run_glue_trainer.py <PARAMETER_NAME> --help``
like in the following::

    >>> python examples/glue/run_glue_trainer.py
    finetune_hparams --help
"""
from argparse import ArgumentParser
import argparse
from collections import defaultdict
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import Pool
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Dict, List, Optional, Tuple, Sequence, Any
from torch.utils.data import DataLoader
import torch
import yahp as hp
import yaml

from composer.trainer.trainer import Trainer
from composer.utils.checkpoint import save_checkpoint
from nlp_trainer_hparams import GLUETrainerHparams, NLPTrainerHparams
from tabulate import tabulate
import multiprocessing
from composer.core.data_spec import DataSpec
from composer.core.time import Time, Timestamp, TimeUnit
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils.file_helpers import format_name_with_dist_and_time
from composer.utils.misc import get_free_tcp_port, warning_on_one_line
from composer.models import ComposerClassifier
from torch.utils.data import Dataset
from composer.models.bert.model import create_bert_classification
from composer.datasets import create_glue_dataset

Metrics = Dict[str, Dict[str, Any]]

import transformers

# squelch fine-tuning warnings
transformers.logging.set_verbosity_error()


class FineTuneJob:
    """Encapsulates a fine-tuning job.

    Tasks should subclass FineTuneJob and implement the
    get_trainer() method.

    Args:
        load_path (str, optional): path to load checkpoints. Default: None
        save_folder (str, optional): path to save checkpoints. Default: None
        kwargs (dict, optional): additional arguments passed available to the Trainer.
    """

    def __init__(self, load_path: str = None, save_folder: Optional[str] = False, **kwargs):
        self.load_path = load_path
        self.save_folder = save_folder
        self.kwargs = kwargs

    def get_trainer(self) -> Trainer:
        """Returns the trainer for the job."""
        raise NotImplementedError

    def print_metrics(self, metrics: Metrics):
        print(f'Results for {self.__class__.__name__}:')
        for eval, metric in metrics.items():
            for metric_name, value in metric.items():
                print(f'{eval}: {metric_name}, {value:.3f}')

    def run(self, gpu_queue: Optional[mp.Queue]=None) -> Dict[str, Any]:
        """Trains the model, optionally pulling a GPU id from the queue.

        Returns a dict of: {
            'checkpoints': <list of saved checkpoints>,
            'metrics': <Dict[dataset_label, Dict[metric_name, result]]>
        }

        """
        gpu_id = gpu_queue.get() if gpu_queue else 0
        torch.cuda.set_device(gpu_id)
        print(f'Running {self.__class__.__name__} on GPU {gpu_id}')

        try:
            trainer = self.get_trainer()
            trainer.fit()

            if trainer.saved_checkpoints:
                saved_checkpoint = trainer.saved_checkpoints
            else:
                saved_checkpoint = None

            collected_metrics = {}
            for eval_name, metrics in trainer.state.eval_metrics.items():
                collected_metrics[eval_name] = {
                    name: metric.compute().cpu().numpy()
                    for name, metric in metrics.items()
                }

            trainer.close()
            self.print_metrics(collected_metrics)
        finally:
            # release the GPU for other jobs
            if gpu_queue:
                print(f'Releasing GPU {gpu_id}')
                gpu_queue.put(gpu_id)

        return {
            'checkpoints': saved_checkpoint,
            'metrics': collected_metrics
        }


def _setup_gpu_queue(num_gpus: int, manager: BaseContext) -> mp.Queue:
    """Returns a queue with [0, 1, .. num_gpus]."""
    gpu_queue = manager.Queue(num_gpus)
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue

def _get_unique_ids(names: List[str]):
    # ['a', 'a', 'c', 'a'] -> ['a_1', 'a_2', 'c', 'a_3']
    counter = defaultdict(int)
    ids = []
    for name in names:
        ids.append(counter[name])
        counter[name] += 1
    return [f'{n}_{i}' for (n, i) in zip(names, ids)]



def run_jobs(jobs: List[FineTuneJob]):
    """Runs a list of jobs across GPUs."""
    num_gpus = torch.cuda.device_count()

    with mp.Manager() as manager:

        # workers get gpu ids from this queue
        # to set the GPU to run on
        gpu_queue = _setup_gpu_queue(num_gpus, manager)

        job_ids = _get_unique_ids([job.__class__.__name__ for job in jobs])

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
            results = [
                pool.apply_async(job.run, args=(gpu_queue,))
                for job in jobs
            ]

            pool.close()
            pool.join()

    finished_results = {
        job_id: r.get()
        for (job_id, r) in zip(job_ids, results)
    }
    return finished_results


def _print_table(results: Dict[str, Dict[str, Any]]):
    """Pretty prints a table given a results dictionary."""

    header = '{job_name:10}| {eval_task:25}| {name:15}|'
    row_format = header + ' {value:.2f}'
    print(header.format(job_name='Job', eval_task='Dataset', name='Metric'))
    print('-' * 55)
    for job_name, result in results.items():
        for eval_task, eval_results in result['metrics'].items():
            for name, metric in eval_results.items():
                print(row_format.format(
                    job_name=job_name.replace('Job', ''),
                    eval_task=eval_task,
                    name=name,
                    value=metric*100,
                ))



def main():
    from glue_jobs import MNLIJob, RTEJob, QQPJob

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', type=str, help='Path to pretrained checkpoint.', default=None)
    parser.add_argument('--wandb_project', type=str, help='wandb project', default=None)
    parser.add_argument('--wandb_entity', type=str, help='wandb entity', default=None)
    parser.add_argument('--run_name', type=str, help='run_name', default=None)
    parser.add_argument('--save_folder', type=str, help='save folder for checkpoints', required=True)

    args = parser.parse_args()

    def get_job_config():
        loggers = []
        if args.wandb_project is not None:
            loggers += [WandBLogger(project=args.wandb_project, entity=args.wandb_entity)]

        return {
            'save_overwrite': True,
            'device': 'gpu',
            'progress_bar': False,
            'log_to_console': False,
            'loggers': loggers,
            'run_name': args.run_name,
            'max_duration': '100ba',
            'load_weights_only': True,
            'load_ignore_keys': ['state/model/model.classifier*'],
        }

    jobs = [
        MNLIJob(save_folder=os.path.join(args.save_folder, 'mnli'), load_path=args.pretrained, **get_job_config()),
        QQPJob(save_folder=None, load_path=args.pretrained, **get_job_config()),
        QQPJob(save_folder=None, load_path=args.pretrained, **get_job_config()),
        QQPJob(save_folder=None, load_path=args.pretrained, **get_job_config()),
    ]

    results = run_jobs(jobs)

    _print_table(results)

    # then finetune from MNLI checkpoint
    mnli_checkpoint = results['MNLIJob_0']['checkpoints'][-1]
    if not os.path.exists(mnli_checkpoint):
        raise FileNotFoundError(
            f'{mnli_checkpoint} missing, likely an error in the MNLI fine-tuning job.'
        )

    load_args = {
        'load_path': mnli_checkpoint,
    }

    jobs = [
        RTEJob(save_folder=None, **load_args, **get_job_config()),
        RTEJob(save_folder=None, **load_args, **get_job_config()),
    ]

    results = run_jobs(jobs)

    _print_table(results)


if __name__ == '__main__':
    main()
