from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional


class Logger:
    """Class for logging LabelModel.

    Parameters
    ----------
    log_train_every
        Number of units at which to log model

    Attributes
    ----------
    log_train_every
        Number of units at which to log model
    unit_count
        Running total of number of units passed without logging
    """

    def __init__(self, log_train_every: int) -> None:
        self.log_train_every = log_train_every
        self.unit_count = 0
        self.total_count = 0

    def check(self) -> bool:
        """Check if the logging frequency has been met.

        Returns
        -------
        bool
            Whether to log or not based on logging frequency
        """
        self.unit_count += 1
        return self.unit_count >= self.log_train_every

    def log(self, metrics_dict: Dict[str, float]) -> None:
        """Print calculated metrics and optionally write to file (json/tb).

        Parameters
        ----------
        metrics_dict
            Dictionary of metric names (keys) and values to log

        Example
        -------
        >>> logger = Logger(log_train_every=5)
        >>> metrics_dict = {"train/loss": 5.00}
        >>> logger.log(metrics_dict)
        [0 epochs]: TRAIN:[loss=5.000]
        """

        self.total_count = self.total_count + self.unit_count
        self.print_to_screen(metrics_dict)
        self.unit_count = 0

    def print_to_screen(self, metrics_dict: Dict[str, float]) -> None:
        """Print all metrics in metrics_dict to screen.

        Parameters
        ----------
        metrics_dict
            Dictionary of metric names (keys) and values to log

        Raises
        ------
        Exception
            If metric names formatted incorrectly

        Example
        -------
        >>> logger = Logger(log_train_every=5)
        >>> metrics_dict = {"train/loss": 5.00}
        >>> logger.print_to_screen(metrics_dict)
        [0 epochs]: TRAIN:[loss=5.000]
        """
        score_strings: DefaultDict[str, List[str]] = defaultdict(list)
        for full_name, value in metrics_dict.items():
            task: Optional[str]
            if full_name.count("/") == 2:
                task, split, metric = full_name.split("/")
            elif full_name.count("/") == 1:
                task = None
                split, metric = full_name.split("/")
            else:
                msg = f"Metric should have form task/split/metric or split/metric, not: {full_name}"
                raise Exception(msg)

            if task:
                metric_name = f"{task}/{metric}"
            else:
                metric_name = metric
            if isinstance(value, float):
                score_strings[split].append(f"{metric_name}={value:0.3f}")
            else:
                score_strings[split].append(f"{metric_name}={value}")

        string = f"[{self.total_count} epochs]:"

        if score_strings["train"]:
            train_scores = f"{', '.join(score_strings['train'])}"
            string += f" TRAIN:[{train_scores}]"
        if score_strings["valid"]:
            valid_scores = f"{', '.join(score_strings['valid'])}"
            string += f" VALID:[{valid_scores}]"
        print(string)
