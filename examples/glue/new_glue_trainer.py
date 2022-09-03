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
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import Pool
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Dict, List, Optional, Tuple, Sequence
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
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.file_helpers import format_name_with_dist_and_time
from composer.utils.misc import get_free_tcp_port, warning_on_one_line
from composer.models import ComposerClassifier
from torch.utils.data import Dataset


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (5, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2


def init_cuda_queue(queue_size: int, ctx: BaseContext) -> mp.Queue:
    """Generate a multiprocessing queue to store queue_size GPU IDs. The multiprocessing package has no way of extracting the worker ID from the worker name; therefore, a queue is used to map pool workers to GPUs to spawn finetune jobs one."""
    cuda_envs = ctx.Queue(queue_size)
    cuda_envs_list = range(queue_size)
    for e in cuda_envs_list:
        cuda_envs.put(e)

    return cuda_envs


def init_cuda_env(cuda_envs: mp.Queue, free_port: int) -> None:
    """Set up a single GPU CUDA environment on initialization of a mp process pool."""
    env = cuda_envs.get()
    torch.cuda.set_device(env)

    # fake a single node world
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_WORLD_SIZE'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'


class FineTuneJob:

    def __init__(self, yaml_path: str = None, load_path: str = None, save_folder: Optional[str] = False, **kwargs):
        self.load_path = load_path
        self.save_folder = save_folder
        self.kwargs = kwargs

    def get_trainer(self):
        return Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            eval_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='5ep',
            load_path=self.load_path,
            load_weights_only=True,
            save_folder=self.save_folder,
            **self.kwargs,
        )

    def get_worker_id(self) -> int:
        # converts SpawnPoolWorker-2 -> 2
        name = multiprocessing.current_process().name
        if 'MainProcess' in name:
            return 0
        return int(name.split('-')[-1])

    def run(self) -> Tuple[Optional[str], float]:
        print(f'Starting on Worker {self.get_worker_id()}')
        trainer = self.get_trainer()
        trainer.fit()

        if trainer.saved_checkpoints:
            saved_checkpoint = trainer.saved_checkpoints
        else:
            saved_checkpoint = None

        metric = trainer.state.eval_metrics['eval']['Accuracy'].compute()
        trainer.close()

        return saved_checkpoint, float(metric)


def run_jobs(jobs: List[FineTuneJob]):

    with Pool(processes=2) as pool:
        results = [pool.apply_async(job.run) for job in jobs]

        pool.close()
        pool.join()

    return [r.get() for r in results]


def main():

    # mock a pretrained checkpoint
    job = FineTuneJob(load_path=None, save_folder='pretrained', save_overwrite=True)
    checkpoints, _ = job.run()

    assert checkpoints is not None
    last_checkpoint = checkpoints[-1]

    jobs = [FineTuneJob(
        load_path=last_checkpoint,
        save_folder=f'test_{k}',
        save_overwrite=True,
    ) for k in range(10)]

    results = run_jobs(jobs)
    import ipdb
    ipdb.set_trace()
    print(results)


#     # Set up CUDA environment(s) and process pool
#     ctx = mp.get_context('spawn')
#     cuda_envs = init_cuda_queue(torch.cuda.device_count(), ctx)
#     free_port = get_free_tcp_port()
#     pool = Pool(processes=torch.cuda.device_count(),
#                 initializer=init_cuda_env,
#                 initargs=(cuda_envs, free_port),
#                 maxtasksperchild=1,
#                 context=ctx)

#     rank = 0
#     # Fine-tune from pre-trained checkpoint(s)
#     ckpt_parent_pairs = zip(ckpt_load_paths, parent_ckpts)
#     for parent_idx, ckpt_parent_pair in enumerate(ckpt_parent_pairs):
#         ckpt_load_path, parent_ckpt = ckpt_parent_pair
#         # `ckpt_load_path` provides the path to the checkpoint from which we load the starting weights used when fine-tuning
#         # `parent_ckpt` keeps track of the original pre-training checkpoint, for tasks with multiple fine-tuning stages (e.g., RTE)
#         # `parent_idx` is used for bookkeeping, so `parent_ckpt` can be internally recovered from the path used to save fine-tune checkpoints
#         for task, save_ckpt in task_to_save_ckpt.items():
#             # Run 1 or more fine-tune trainers from this checkpoint, using a different seed override for each
#             for seed in seed_overrides.get(task, [None]):
#                 pool.apply_async(train_finetune,
#                                  args=(base_yaml_file, task, save_ckpt, ckpt_load_path, parent_ckpt, parent_idx,
#                                        ckpt_save_folder, save_locally, load_locally, free_port + rank, load_ignore_keys,
#                                        seed),
#                                  callback=partial(ingest_finetuning_result,
#                                                   cuda_queue=cuda_envs,
#                                                   task=task,
#                                                   ckpt_filename=parent_ckpt,
#                                                   glue_metrics=None))

#                 rank += 1

# def merge_hparams(hparams: TrainerHparams, override_hparams: GLUETrainerHparams) -> TrainerHparams:
#     """Overrides the atttributes of the hparams instance with those of the provided override_hparams."""
#     hparams.algorithms = override_hparams.algorithms if override_hparams.algorithms else hparams.algorithms
#     hparams.load_ignore_keys = override_hparams.load_ignore_keys if override_hparams.load_ignore_keys else hparams.load_ignore_keys
#     hparams.load_path = override_hparams.load_path if override_hparams.load_path else hparams.load_path
#     hparams.load_object_store = override_hparams.load_object_store if override_hparams.load_object_store else hparams.load_object_store
#     hparams.load_logger_destination = override_hparams.load_logger_destination if override_hparams.load_logger_destination else hparams.load_logger_destination
#     hparams.loggers = override_hparams.loggers if override_hparams.loggers else hparams.loggers
#     hparams.model = override_hparams.model if override_hparams.model else hparams.model
#     hparams.run_name = override_hparams.run_name if override_hparams.run_name else hparams.run_name
#     hparams.save_folder = override_hparams.save_folder if override_hparams.save_folder else hparams.save_folder

#     return hparams

# def spawn_finetuning_jobs(
#     task_to_save_ckpt: Dict[str, bool],
#     ckpt_load_path: str,
#     ckpt_save_folder: str,
#     glue_metrics: GlueState,
#     base_yaml_file: str,
#     save_locally: bool,
#     load_locally: bool,
#     parent_ckpt: Optional[str] = None,
#     load_ignore_keys: Optional[List[str]] = None,
# ) -> None:
#     """Set up CUDA environment and process pool for given finetuning jobs."""
#     ctx = mp.get_context('spawn')
#     cuda_envs = init_cuda_queue(torch.cuda.device_count(), ctx)
#     finetune_tasks = list(task_to_save_ckpt.keys())
#     num_tasks = len(finetune_tasks)

#     if parent_ckpt:
#         wandb_group_name = parent_ckpt
#         logged_ckpt_name = parent_ckpt
#     else:
#         wandb_group_name = ckpt_load_path
#         logged_ckpt_name = ckpt_load_path

#     # finetuning from pretrained checkpoint
#     print(f'FINETUNING ON {ckpt_load_path}!')
#     done_callback = lambda future: log_metrics(
#         metric=future.result(), ckpt_filename=logged_ckpt_name, glue_metrics=glue_metrics)
#     free_port = get_free_tcp_port()
#     executor = Pool(max_workers=torch.cuda.device_count(),
#                     initializer=init_cuda_env,
#                     initargs=(cuda_envs, free_port),
#                     mp_context=ctx)
#     for rank in range(num_tasks):
#         task = finetune_tasks[rank]
#         future = executor.submit(train_finetune, base_yaml_file, task, task_to_save_ckpt[task], ckpt_load_path,
#                                  wandb_group_name, ckpt_save_folder, save_locally, load_locally, free_port + rank,
#                                  load_ignore_keys)
#         future.add_done_callback(done_callback)

#     executor.shutdown(wait=True)  # wait for processes and callbacks to complete

#     cuda_envs.close()
#     cuda_envs.join_thread()

# def train_finetune(
#     base_yaml_file: str,
#     task: str,
#     save_ckpt: bool,
#     load_path: str,
#     wandb_group_name: str,
#     save_folder: str,
#     save_locally: bool,
#     load_locally: bool,
#     master_port: int,
#     load_ignore_keys: Optional[List[str]] = None,
# ):
#     """Run single instance of a finetuning job on given task."""
#     os.environ['MASTER_PORT'] = f'{master_port}'  # set unique master port for each spawn

#     finetune_hparams = NLPTrainerHparams.create(cli_args=False, f=base_yaml_file).finetune_hparams
#     task_hparams = TrainerHparams.create(cli_args=False, f=f'./composer/yamls/models/glue/{task}.yaml')

#     if finetune_hparams:
#         ft_hparams = merge_hparams(task_hparams, finetune_hparams)
#     else:
#         ft_hparams = task_hparams

#     ft_hparams.load_path = load_path
#     ft_hparams.device = DeviceGPU(
#         torch.cuda.current_device())  # set device manually to force finetuning to happen on one GPU
#     ft_hparams.log_to_console = False
#     ft_hparams.progress_bar = False
#     ft_hparams.save_overwrite = True

#     if ft_hparams.load_ignore_keys:
#         assert load_ignore_keys is not None
#         ft_hparams.load_ignore_keys.extend(load_ignore_keys)
#     else:
#         ft_hparams.load_ignore_keys = load_ignore_keys

#     # add finetune-specific tags to wandb if logger exists
#     if ft_hparams.loggers:
#         for logger in ft_hparams.loggers:
#             if isinstance(logger, WandBLogger):
#                 if 'tags' not in logger._init_kwargs.keys():
#                     logger._init_kwargs['tags'] = []
#                 logger._init_kwargs['tags'].append(task)

#     if load_locally:
#         ft_hparams.load_object_store = None

#     # saving single checkpoint at the end of training the task
#     if save_ckpt:
#         # add task specific artifact logging information
#         ft_hparams.save_folder = f'{save_folder}/{task}'
#         save_artifact_name = f'{save_folder}/{task}/ep{{epoch}}-ba{{batch}}-rank{{rank}}'  # ignored if not uploading
#         save_latest_artifact_name = f'{save_folder}/{task}/latest-rank{{rank}}'
#         ft_hparams.save_artifact_name = save_artifact_name
#         ft_hparams.save_latest_artifact_name = save_latest_artifact_name

#         if save_locally:
#             if not os.path.exists(ft_hparams.save_folder):
#                 os.makedirs(ft_hparams.save_folder)

#     print(f'\n --------\n SPAWNING TASK {task.upper()}\n DEVICE: {torch.cuda.current_device()}\n --------')

#     trainer = ft_hparams.initialize_object()

#     # if using wandb, store the config and other information inside the wandb run
#     try:
#         import wandb
#     except ImportError:
#         pass
#     else:
#         if wandb.run is not None:
#             wandb.config.update(ft_hparams.to_dict())
#             wandb.config.update({'pretrained_ckpt': wandb_group_name, 'task': task})

#     trainer.fit()
#     print(f'\nFINISHED TRAINING TASK {task.upper()}\n')
#     # recursively move metrics to CPU to avoid pickling issues
#     return DeviceCPU().batch_to_device(trainer.state.eval_metrics)

# def get_args() -> str:
#     """Get NLPTrainerHparams arguments from CLI."""
#     parser = hp.get_argparse(NLPTrainerHparams)
#     args, _ = parser.parse_known_args()
#     return args.file

# def validate_args(hp: NLPTrainerHparams) -> None:
#     """Validate CLI args as well as finetune-specific parameters."""
#     if hp.training_scheme not in ('finetune', 'pretrain', 'all'):
#         raise ValueError('training_scheme must be one of "finetune", "pretrain," or "all"')

#     if hp.training_scheme != 'finetune' and not hp.pretrain_hparams:
#         raise ValueError('pretrain_hparams must be specified if pretraining a model')

#     elif hp.training_scheme == 'finetune' and ((not hp.finetune_hparams) or
#                                                (hp.finetune_hparams and not hp.finetune_hparams.finetune_ckpts)):
#         raise ValueError('load_path to checkpoints must be specified if finetuning a model')

#     elif hp.training_scheme == 'pretrain' and hp.finetune_hparams:
#         warnings.warn('finetune_hparams specified. These values will be ignored during pretraining.')

#     elif hp.training_scheme == 'all' and hp.finetune_hparams is None:
#         warnings.warn('No shared finetune_hparams specified. All finetune tasks will use their default configurations.')

#     elif hp.training_scheme == 'all' and hp.finetune_hparams and hp.finetune_hparams.finetune_ckpts:
#         warnings.warn('finetune_ckpts specified in finetune_hparams. This value will be overriden during finetuning.')

# def get_finetune_hparams() -> Tuple[GLUETrainerHparams, str, bool, bool]:
#     """Extract finetune-specific hparams from the provided file and add entrypoint specific args to it."""
#     hp = NLPTrainerHparams.create()
#     validate_args(hp)

#     training_scheme = hp.training_scheme

#     save_locally = True
#     load_locally = True
#     hparams = GLUETrainerHparams(model=None)
#     if training_scheme in ('finetune', 'all'):
#         if hp.finetune_hparams:
#             hparams = hp.finetune_hparams
#             if hparams.load_object_store:
#                 load_locally = False
#             if hparams.loggers:
#                 for l in hparams.loggers:
#                     if isinstance(l, ObjectStoreLogger):
#                         save_locally = False
#                     if isinstance(l, WandBLogger) and l._log_artifacts:
#                         save_locally = False

#     return hparams, training_scheme, save_locally, load_locally

# def get_ckpt_names(hp: TrainerHparams, run_name: str, dataloader_len: int) -> List[str]:
#     """Extract list of checkpoints that will be saved by the given configuration."""
#     ckpt_names = []
#     assert hp.save_interval is not None
#     assert hp.max_duration is not None
#     interval = Time.from_timestring(str(hp.save_interval))
#     duration = Time.from_timestring(str(hp.max_duration))

#     ep = 0
#     ba = 0
#     loop = True
#     save = False
#     save_last_batch = False
#     while loop:
#         if save:
#             time = Timestamp(epoch=ep, batch=ba)
#             formatted_ckpt_name = format_name_with_dist_and_time(hp.save_artifact_name, run_name, time)
#             ckpt_names.append(formatted_ckpt_name)
#             save = False

#         ba += interval.value
#         if interval.unit == TimeUnit.BATCH:
#             save = True
#         if ba >= dataloader_len:  # batches per epoch
#             ep += 1
#             if interval.unit == TimeUnit.EPOCH:
#                 save = True

#         if duration.unit == TimeUnit.BATCH:
#             if ba >= duration.value:
#                 loop = False
#                 save_last_batch = True
#                 if ba > duration.value:
#                     ba = duration.value
#         elif duration.unit == TimeUnit.EPOCH:
#             if ep >= duration.value:
#                 loop = False
#         elif duration.unit == TimeUnit.SAMPLE:
#             if ba * hp.train_batch_size >= duration.value:
#                 loop = False
#                 save_last_batch = True
#                 if ba * hp.train_batch_size > duration.value:
#                     ba = duration.value // hp.train_batch_size

#     # save very last batch if incrementing batches passed it
#     if save_last_batch:
#         time = Timestamp(epoch=ep, batch=ba)
#         formatted_ckpt_name = format_name_with_dist_and_time(hp.save_artifact_name, run_name, time)
#         ckpt_names.append(formatted_ckpt_name)

#     return ckpt_names

# def run_pretrainer(training_scheme: str, file: str, finetune_hparams: GLUETrainerHparams) -> None:
#     """Logic for handling a pretraining job spawn based on storage and training settings."""
#     root_dir = os.path.join(os.path.dirname(__file__), '..')
#     training_script = os.path.join(root_dir, 'run_composer_trainer.py')

#     # manually copy pretrain_hparams to temporary file
#     tmp_dir = tempfile.TemporaryDirectory()
#     tmp_file = os.path.join(tmp_dir.name, 'pretrained_hparams.yaml')
#     with open(file) as infile, open(tmp_file, 'w+') as outfile:
#         hparams = yaml.load(infile, yaml.Loader)
#         pretrain_hparams = hparams['pretrain_hparams']
#         yaml.dump(pretrain_hparams, outfile)

#     hp = TrainerHparams.create(cli_args=False, f=tmp_file)
#     assert hp.train_dataset is not None
#     assert hp.train_batch_size is not None
#     dataloader = hp.train_dataset.initialize_object(dataloader_hparams=hp.dataloader, batch_size=hp.train_batch_size)
#     assert isinstance(dataloader, DataSpec)
#     dataloader_len = len(dataloader.dataloader)  # type: ignore
#     run_name = hp.run_name
#     assert run_name is not None
#     assert hp.save_folder is not None
#     save_folder = os.path.join(run_name, hp.save_folder)

#     if training_scheme == 'all':  # extract run_name from trainer args for finetuning
#         # list and save checkpoint paths
#         finetune_hparams.save_folder = save_folder
#         finetune_hparams.finetune_ckpts = get_ckpt_names(hp, run_name, dataloader_len)

#     # call via composer to ensure pretraining done distributedly across all available GPUs
#     subprocess.run(args=['composer', training_script, '-f', tmp_file, '--save_folder', save_folder], check=True)

# def run_finetuner(training_scheme: str, file: str, save_locally: bool, load_locally: bool, save_folder: str,
#                   finetune_hparams, glue_metrics: GlueState) -> None:
#     """Logic for handling a finetuning job spawn based on storage and training settings."""
#     # set automatic load and save paths
#     if load_locally:
#         all_ckpts_list = os.listdir(save_folder)
#     else:
#         all_ckpts_list = finetune_hparams.finetune_ckpts

#     # finetune on every pretrained checkpoint
#     for ckpt_filename in all_ckpts_list:
#         parent_ckpt = ckpt_filename  # necessary for logging

#         # TODO (Alex): Remove two-step finetuning setup after CO-806 is resolved
#         task_to_save_ckpt = {'cola': False, 'sst-2': False, 'qqp': False, 'qnli': False, 'mnli': True}
#         spawn_finetuning_jobs(task_to_save_ckpt,
#                               ckpt_filename,
#                               save_folder,
#                               glue_metrics,
#                               file,
#                               save_locally,
#                               load_locally,
#                               parent_ckpt=parent_ckpt,
#                               load_ignore_keys=['state/model/model.classifier*'])

#         # finetune on inference tasks using last mnli checkpoint
#         ckpt_filename = f'{save_folder}/mnli/latest-rank0'

#         mnli_task_to_save_ckpt = {'rte': False, 'mrpc': False, 'stsb': False}
#         # delete finetuning head to reinitialize number of classes
#         spawn_finetuning_jobs(
#             mnli_task_to_save_ckpt,
#             ckpt_filename,
#             save_folder,
#             glue_metrics,
#             file,
#             save_locally,
#             load_locally=save_locally,  # if mnli saved ckpts locally, load locally
#             parent_ckpt=parent_ckpt,
#             load_ignore_keys=['state/model/model.classifier*'],
#         )

# def _main() -> None:
#     warnings.formatwarning = warning_on_one_line

#     if len(sys.argv) == 1:
#         sys.argv = [sys.argv[0], '--help']

#     file = get_args()
#     finetune_hparams, training_scheme, save_locally, load_locally = get_finetune_hparams()

#     # Pretrain
#     if training_scheme in ('pretrain', 'all'):
#         run_pretrainer(training_scheme, file, finetune_hparams)
#         print('PRETRAINING COMPLETE')

#     # Finetune
#     glue_task_names = ['cola', 'sst2', 'qqp', 'qnli', 'mnli', 'rte', 'mrpc', 'stsb']
#     glue_metrics = GlueState(glue_task_names, {})
#     if training_scheme in ('finetune', 'all'):
#         assert finetune_hparams.save_folder is not None
#         run_finetuner(training_scheme, file, save_locally, load_locally, finetune_hparams.save_folder, finetune_hparams,
#                       glue_metrics)
#         print('FINETUNING COMPLETE')

#         # output GLUE metrics
#         print_metrics(glue_metrics)

if __name__ == '__main__':
    main()
