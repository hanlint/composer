from new_glue_trainer import FineTuneJob
from composer.models.bert.model import create_bert_classification
from composer.datasets import create_glue_dataset
from composer.core.evaluator import Evaluator
from composer.optim.decoupled_weight_decay import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler
from composer.trainer.trainer import Trainer
from torch.utils.data import DataLoader


class MNLIJob(FineTuneJob):

    num_labels = 3
    eval_interval = '2300ba'

    def get_trainer(self):
        dataset_kwargs = {
            'task': 'mnli',
            'tokenizer_name': 'bert-base-uncased',
            'max_seq_length': 256,
        }

        dataloader_kwargs = {
            'batch_size': 48,
            'num_workers': 8,
            'shuffle': False,
            'drop_last': False,
        }

        model = create_bert_classification(
            num_labels=self.num_labels,
            pretrained_model_name='bert-base-uncased',
            use_pretrained=True,
        )
        optimizer = DecoupledAdamW(
            model.parameters(),
            lr=5.0e-5,
            beats=(0.9, 0.98),
            eps=1.0e-6,
            weight_decay=5.0e-6,
        )

        scheduler = LinearWithWarmupScheduler(t_warmup='0.06dur')

        train_dataset = create_glue_dataset(split='train', **dataset_kwargs),
        eval_mnli = create_glue_dataset(split='validation_matched', **dataset_kwargs)
        eval_mnli_mismatch = create_glue_dataset(split='validation_mismatched', **dataset_kwargs)

        return Trainer(
            model=model,
            optimizers=optimizer,
            schedulers=scheduler,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                **dataloader_kwargs,
            ),
            evaluators=[
                Evaluator(
                    label='glue_mnli',
                    dataloader=DataLoader(
                        dataset=eval_mnli,
                        **dataloader_kwargs,
                    ),
                    metric_names=['Accuracy'],
                ),
                Evaluator(
                    label='glue_mnli_mismatched',
                    dataloader=DataLoader(
                        dataset=eval_mnli_mismatch,
                        **dataloader_kwargs,
                    ),
                    metric_names=['Accuracy'],
                ),
            ],
            eval_interval=self.eval_interval,
            load_path=self.load_path,
            save_folder=self.save_folder,
            load_weights_only=True,
            **self.kwargs,
        )


class RTEJob(FineTuneJob):

    num_labels = 2
    eval_interval = '1000ba'

    def get_trainer(self):
        dataset_kwargs = {
            'task': 'rte',
            'tokenizer_name': 'bert-base-uncased',
            'max_seq_length': 256,
        }

        dataloader_kwargs = {
            'batch_size': 16,
            'num_workers': 8,
            'shuffle': False,
            'drop_last': False,
        }

        model = create_bert_classification(
            num_labels=self.num_labels,
            pretrained_model_name='bert-base-uncased',
            use_pretrained=True,
        )
        optimizer = DecoupledAdamW(
            model.parameters(),
            lr=1.0e-5,
            beats=(0.9, 0.98),
            eps=1.0e-6,
            weight_decay=1.0e-6,
        )

        scheduler = LinearWithWarmupScheduler(t_warmup='0.06dur')

        train_dataset = create_glue_dataset(split='train', **dataset_kwargs),
        eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)

        return Trainer(
            model=model,
            optimizers=optimizer,
            schedulers=scheduler,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                **dataloader_kwargs,
            ),
            evaluators=[
                Evaluator(
                    label='glue_rte',
                    dataloader=DataLoader(
                        dataset=eval_dataset,
                        **dataloader_kwargs,
                    ),
                    metric_names=['Accuracy'],
                ),
            ],
            eval_interval=self.eval_interval,
            load_path=self.load_path,
            save_folder=self.save_folder,
            load_weights_only=True,
            **self.kwargs,
        )


class QQPJob(FineTuneJob):

    num_labels = 2
    eval_interval = '2000ba'

    def get_trainer(self):
        dataset_kwargs = {
            'task': 'rte',
            'tokenizer_name': 'bert-base-uncased',
            'max_seq_length': 256,
        }

        dataloader_kwargs = {
            'batch_size': 16,
            'num_workers': 8,
            'shuffle': False,
            'drop_last': False,
        }

        model = create_bert_classification(
            num_labels=self.num_labels,
            pretrained_model_name='bert-base-uncased',
            use_pretrained=True,
        )
        optimizer = DecoupledAdamW(
            model.parameters(),
            lr=3.0e-5,
            beats=(0.9, 0.98),
            eps=1.0e-6,
            weight_decay=3.0e-6,
        )

        scheduler = LinearWithWarmupScheduler(t_warmup='0.06dur')

        train_dataset = create_glue_dataset(split='train', **dataset_kwargs),
        eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)

        return Trainer(
            model=model,
            optimizers=optimizer,
            schedulers=scheduler,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                **dataloader_kwargs,
            ),
            evaluators=[
                Evaluator(
                    label='glue_rte',
                    dataloader=DataLoader(
                        dataset=eval_dataset,
                        **dataloader_kwargs,
                    ),
                    metric_names=['Accuracy'],
                ),
            ],
            eval_interval=self.eval_interval,
            load_path=self.load_path,
            save_folder=self.save_folder,
            load_weights_only=True,
            **self.kwargs,
        )
