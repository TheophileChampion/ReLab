from typing import Any, Union, Dict, TypeVar, Tuple, SupportsFloat

import numpy
import torch

Config = Dict[str, Any]  # An alias representing a configuration.

# An alias representing some config information either:
# - the full configuration, or
# - the value associated to a specific key.
ConfigInfo = Union[Config, Any]

Device = Any  # An alias representing a torch device.
Parameter = torch.nn.parameter.Parameter  # An alias representing a torch parameter.
Loss = Any  # An alias representing a torch loss like MSELoss.
Checkpoint = Any  # An alias representing a torch checkpoint returned by torch.load.
DataType = Any  # An alias representing the type of tensor elements.

ActionType = Union[torch.Tensor, numpy.ndarray, int]  # An alias representing an action.

# An alias representing the return of (Gym) environment's step function, i.e. a tuple containing:
# - an observation
# - a reward
# - a boolean indicating whether the episode terminated
# - a boolean indicating whether the episode was truncated
# - a dictionary containing additional information
GymStepData = Tuple[numpy.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]

T = TypeVar("T")
ScalarOrTuple = Union[T, tuple[T, T]]
