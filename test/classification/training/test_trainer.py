import copy
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from snorkel.classification.data import DictDataLoader, DictDataset
from snorkel.classification.scorer import Scorer
from snorkel.classification.snorkel_classifier import Operation, SnorkelClassifier, Task
from snorkel.classification.training import Trainer
from snorkel.classification.training.loggers import LogWriter, TensorBoardWriter

TASK_NAMES = ["task1", "task2"]
base_config = {"n_epochs": 1, "progress_bar": False}
NUM_EXAMPLES = 6
BATCH_SIZE = 2
BATCHES_PER_EPOCH = NUM_EXAMPLES / BATCH_SIZE


def create_dataloader(task_name="task", split="train"):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES, 1).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
            module2_name: nn.Linear(10, 2),
        }
    )

    op1 = Operation(module_name=module1_name, inputs=[("_input_", "data")])
    op2 = Operation(module_name=module2_name, inputs=[(op1.name, 0)])

    task_flow = [op1, op2]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


dataloaders = [create_dataloader(task_name) for task_name in TASK_NAMES]
tasks = [
    create_task(TASK_NAMES[0], module_suffixes=["A", "A"]),
    create_task(TASK_NAMES[1], module_suffixes=["A", "B"]),
]
model = SnorkelClassifier([tasks[0]])


class TrainerTest(unittest.TestCase):
    def test_trainer_onetask(self):
        """Train a single-task model"""
        trainer = Trainer(**base_config)
        trainer.train_model(model, [dataloaders[0]])

    def test_trainer_twotask(self):
        """Train a model with overlapping modules and flows"""
        multitask_model = SnorkelClassifier(tasks)
        trainer = Trainer(**base_config)
        trainer.train_model(multitask_model, dataloaders)

    def test_trainer_errors(self):
        dataloader = copy.deepcopy(dataloaders[0])

        # No train split
        trainer = Trainer(**base_config)
        dataloader.dataset.split = "valid"
        with self.assertRaisesRegex(ValueError, "Cannot find any dataloaders"):
            trainer.train_model(model, [dataloader])

        # Unused split
        trainer = Trainer(**base_config, valid_split="val")
        with self.assertRaisesRegex(ValueError, "Dataloader splits must be"):
            trainer.train_model(model, [dataloader])

    def test_checkpointer_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            more_config = {
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_dir": None},
                "log_writer_config": {"log_dir": temp_dir},
            }
            trainer = Trainer(**base_config, **more_config, logging=True)
            trainer.train_model(model, [dataloaders[0]])
            self.assertIsNotNone(trainer.checkpointer)
            self.assertEqual(
                trainer.checkpointer.checkpoint_dir, trainer.log_writer.log_dir
            )

            with self.assertRaisesRegex(ValueError, "Checkpointing is on but"):
                trainer = Trainer(**base_config, **more_config, logging=False)
                trainer.train_model(model, [dataloaders[0]])

    def test_log_writer_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_writer_config = {"log_dir": temp_dir}
            trainer = Trainer(
                **base_config,
                logging=True,
                log_writer="json",
                log_writer_config=log_writer_config,
            )
            trainer.train_model(model, [dataloaders[0]])
            self.assertIsInstance(trainer.log_writer, LogWriter)

            log_writer_config = {"log_dir": temp_dir}
            trainer = Trainer(
                **base_config,
                logging=True,
                log_writer="tensorboard",
                log_writer_config=log_writer_config,
            )
            trainer.train_model(model, [dataloaders[0]])
            self.assertIsInstance(trainer.log_writer, TensorBoardWriter)

            log_writer_config = {"log_dir": temp_dir}
            with self.assertRaisesRegex(ValueError, "Unrecognized writer"):
                trainer = Trainer(
                    **base_config,
                    logging=True,
                    log_writer="foo",
                    log_writer_config=log_writer_config,
                )
                trainer.train_model(model, [dataloaders[0]])

    def test_optimizer_init(self):
        trainer = Trainer(**base_config, optimizer="sgd")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.SGD)

        trainer = Trainer(**base_config, optimizer="adam")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.Adam)

        trainer = Trainer(**base_config, optimizer="adamax")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.Adamax)

        with self.assertRaisesRegex(ValueError, "Unrecognized optimizer"):
            trainer = Trainer(**base_config, optimizer="foo")
            trainer.train_model(model, [dataloaders[0]])

    def test_scheduler_init(self):
        trainer = Trainer(**base_config, lr_scheduler="constant")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsNone(trainer.lr_scheduler)

        trainer = Trainer(**base_config, lr_scheduler="linear")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.LambdaLR)

        trainer = Trainer(**base_config, lr_scheduler="exponential")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.ExponentialLR)

        trainer = Trainer(**base_config, lr_scheduler="step")
        trainer.train_model(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.StepLR)

        with self.assertRaisesRegex(ValueError, "Unrecognized lr scheduler"):
            trainer = Trainer(**base_config, lr_scheduler="foo")
            trainer.train_model(model, [dataloaders[0]])

    def test_warmup(self):
        lr_scheduler_config = {"warmup_steps": 1, "warmup_unit": "batches"}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.train_model(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, 1)

        lr_scheduler_config = {"warmup_steps": 1, "warmup_unit": "epochs"}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.train_model(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, BATCHES_PER_EPOCH)

        lr_scheduler_config = {"warmup_percentage": 1 / BATCHES_PER_EPOCH}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.train_model(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, 1)


if __name__ == "__main__":
    unittest.main()
