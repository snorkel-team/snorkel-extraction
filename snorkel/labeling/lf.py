from typing import Any, Callable, List, Mapping, Optional, Tuple

from snorkel.types import DataPoint

from .preprocess import BasePreprocessor, PreprocessorMode


class LabelingFunction:
    """Base object for labeling functions.

    A labeling function (LF) is a function that takes a data point
    as input and produces an integer label, corresponding to a
    class. A labeling function can also abstain from voting by
    outputting 0. For examples, see the Snorkel tutorials.

    This object wraps a Python function outputting a label. Metadata
    about the input data types and label space are stored. Extra
    functionality, such as running preprocessors and storing
    resources, is provided. Simple LFs can be defined via a
    decorator. See `labeling_function`.

    Parameters
    ----------
    name
        Name of the LF
    f
        Function that implements the core LF logic
    label_space
        Set of labels the LF can output, including 0
    schema
        Fields of the input `DataPoint`s the LF accesses
    resources
        Labeling resources passed in to `f` via `kwargs`
    preprocessors
        Preprocessors to run on data points before LF execution
    fault_tolerant
        Output 0 if LF execution fails?

    Attributes
    ----------
    name
        See above
    label_space
        See above
    schema
        See above
    fault_tolerant
        See above

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors
    """

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        self.name = name
        self.label_space = label_space
        self.schema = schema
        self.fault_tolerant = fault_tolerant
        self._f = f
        self._resources = resources or {}
        self._preprocessors = preprocessors or []

    def set_fault_tolerant(self, fault_tolerant: bool = True) -> None:
        """Change LF fault tolerance mode.

        If set to fault tolerant mode, the LF returns 0 if
        an exception is raised during execution.

        Parameters
        ----------
        fault_tolerant
            Set to fault tolerant mode?
        """
        self.fault_tolerant = fault_tolerant

    def set_preprocessor_mode(self, mode: PreprocessorMode) -> None:
        """Change LF preprocessor mode.

        Parameters
        ----------
        mode
            Mode to set preprocessors to
        """
        for preprocessor in self._preprocessors:
            preprocessor.set_mode(mode)

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        for preprocessor in self._preprocessors:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint) -> int:
        """Label data point.

        Runs all preprocessors, then passes to LF. If an exception
        is encountered and the LF is in fault tolerant mode,
        the LF abstains from voting.

        Parameters
        ----------
        x
            Data point to label

        Returns
        -------
        int
            Label for data point
        """
        x = self._preprocess_data_point(x)
        if self.fault_tolerant:
            try:
                return self._f(x, **self._resources)
            except Exception:
                return 0
        return self._f(x, **self._resources)

    def __repr__(self) -> str:
        schema_str = f", DataPoint schema: {self.schema}" if self.schema else ""
        label_str = f", Label space: {self.label_space}" if self.label_space else ""
        preprocessor_str = f", Preprocessors: {self._preprocessors}"
        return f"Labeling function {self.name}{schema_str}{label_str}{preprocessor_str}"


class labeling_function:
    """Decorator to define a LabelingFunction object from a function.

    Parameters
    ----------
    name
        Name of the LF. If None, uses the name of the wrapped function.
    label_space
        Set of labels the LF can output, including 0
    schema
        Fields of the input `DataPoint`s the LF accesses
    resources
        Labeling resources passed in to `f` via `kwargs`
    preprocessors
        Preprocessors to run on data points before LF execution
    fault_tolerant
        Output 0 if LF execution fails?

    Examples
    --------
    ```
    @labeling_function()
    def f(x: DataPoint) -> int:
        return 1 if x.a > 42 else 0
    print(f)  # "Labeling function f"

    @labeling_function(name="my_lf")
    def g(x: DataPoint) -> int:
        return 1 if x.a > 42 else 0
    print(g)  # "Labeling function my_lf"
    ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        self.name = name
        self.label_space = label_space
        self.schema = schema
        self.resources = resources
        self.preprocessors = preprocessors
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        """Wrap a function to create a `LabelingFunction`.

        Parameters
        ----------
        f
            Function that implements the core LF logic

        Returns
        -------
        LabelingFunction
            New `LabelingFunction` executing logic in wrapped function
        """
        name = self.name or f.__name__
        return LabelingFunction(
            name=name,
            f=f,
            label_space=self.label_space,
            schema=self.schema,
            resources=self.resources,
            preprocessors=self.preprocessors,
            fault_tolerant=self.fault_tolerant,
        )
