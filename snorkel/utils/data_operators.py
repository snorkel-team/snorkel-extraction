from collections import Counter
from typing import Any, Callable, List, Mapping, Optional

from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM


def check_unique_names(names: List[str]) -> None:
    """Check that operator names are unique."""
    k, ct = Counter(names).most_common(1)[0]
    if ct > 1:
        raise ValueError(f"Operator names not unique: {ct} operators with name {k}")


class OperatorDecorator:
    """Decorator to define a Snorkel Operator (e.g. LabelingFunction).

    Parameters
    ----------
    operator
        Operator class (e.g. LabelingFunction) that will be initialized by decorator
    name
        Name of the operator
    resources
        Resources passed in to ``f`` via ``kwargs``
    preprocessors
        Preprocessors to run on data points before operator execution
    fault_tolerant
        Output ``-1`` if LF execution fails?
    """

    def __init__(
        self,
        operator: Callable,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:

        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")

        self._operator = operator
        self.name = name
        self.resources = resources
        self.pre = pre
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., int]) -> Callable:
        """Wrap a function to create a Snorkel Operator.

        Parameters
        ----------
        f
        Function that implements the core LF logic

        Returns
        -------
        Callable
            New ``Operator`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return self._operator(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            fault_tolerant=self.fault_tolerant,
        )


class NLPOperatorDecorator(OperatorDecorator):
    """Decorator to define an NLPLabelingFunction object from a function.

    Parameters
    ----------
    name
        See ``OperatorDecorator``
    resources
        See ``OperatorDecorator``
    preprocessors
        See ``OperatorDecorator``
    fault_tolerant
        See ``OperatorDecorator``
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        SpaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?
    """

    def __init__(
        self,
        operator: Callable,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
        text_field: str = "text",
        doc_field: str = "doc",
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
    ) -> None:

        super().__init__(operator, name, resources, pre, fault_tolerant)

        self.text_field = text_field
        self.doc_field = doc_field
        self.language = language
        self.disable = disable
        self.memoize = memoize

    def __call__(self, f: Callable[..., int]) -> Callable:
        """Wrap a function to create an ``NLPLabelingFunction``.

        Parameters
        ----------
        f
            Function that implements the core NLP LF logic

        Returns
        -------
        NLPLabelingFunction
            New ``NLPLabelingFunction`` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return self._operator(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            fault_tolerant=self.fault_tolerant,
            text_field=self.text_field,
            doc_field=self.doc_field,
            language=self.language,
            disable=self.disable,
            memoize=self.memoize,
        )
