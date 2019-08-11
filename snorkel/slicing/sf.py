from typing import Any, List, Mapping, Optional

from snorkel.labeling import LabelingFunction
from snorkel.preprocess import BasePreprocessor
from snorkel.utils.data_operators import OperatorDecorator


class SlicingFunction(LabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    pass


class slicing_function(OperatorDecorator):
    """Decorator to define a SlicingFunction object from a function.

    Parameters
    ----------
    name
        See ``snorkel.utils.OperatorDecorator``.
    resources
        See ``snorkel.utils.OperatorDecorator``.
    preprocessors
        See ``snorkel.utils.OperatorDecorator``.
    fault_tolerant
        See ``snorkel.utils.OperatorDecorator``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        super().__init__(
            operator=SlicingFunction,
            name=name,
            resources=resources,
            pre=pre,
            fault_tolerant=fault_tolerant,
        )
