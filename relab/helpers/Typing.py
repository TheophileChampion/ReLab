from typing import Any, Dict, KeysView, List, SupportsFloat, Tuple, TypeVar, Union

import numpy
import torch

# An alias representing a configuration.
Config = Dict[str, Any]

# An alias representing some config information either:
# - the full configuration, or
# - the value associated to a specific key.
ConfigInfo = Union[Config, Any]

# An alias representing a list of agent's attribute names.
AttributeNames = Union[List[str], KeysView]

# An alias representing a torch device.
Device = Any
# An alias representing a torch parameter.
Parameter = torch.nn.parameter.Parameter
# An alias representing a torch loss like MSELoss.
Loss = Any
# An alias representing a torch checkpoint returned by torch.load.
Checkpoint = Any
# An alias representing the type of tensor elements.
DataType = Any
# An alias representing a torch optimizer.
Optimizer = Any

# An alias representing an action.
ActionType = Union[torch.Tensor, numpy.ndarray, int]
# An alias representing an observation.
ObservationType = Union[torch.Tensor, numpy.ndarray]

# An alias representing the return of (Gym) environment's step function, i.e. a tuple containing:
# - an observation
# - a reward
# - a boolean indicating whether the episode terminated
# - a boolean indicating whether the episode was truncated
# - a dictionary containing additional information
GymStepData = Tuple[ObservationType, SupportsFloat, bool, bool, Dict[str, Any]]

# An alias representing a batch of experiences.
Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# An alias representing either a scalar or a tuple.
T = TypeVar("T")
ScalarOrTuple = Union[T, tuple[T, T]]
