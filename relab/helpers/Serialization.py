import logging
from typing import Any, List

from relab.helpers.Typing import Checkpoint, Optimizer
from torch import optim
from torch.nn import Module


def safe_load(checkpoint: Checkpoint, key: str) -> Any:
    """!
    Load the value corresponding to the key in the checkpoint.
    @param checkpoint: the checkpoint
    @param key: the key
    @return the value, or None if the key is not in the checkpoint
    """
    if key not in checkpoint.keys():
        return None
    return checkpoint[key]


def safe_load_state_dict(obj: Any, checkpoint: Checkpoint, key: str) -> None:
    """
    Load the state dictionary of an object from a checkpoint
    :param obj: the object whose dictionary must be loaded
    :param checkpoint: the checkpoint
    :param key: the key in the checkpoint where the dictionary is located
    """
    if checkpoint is None:
        return
    if key not in checkpoint.keys():
        logging.info(f"Could not load state_dict of {key} from checkpoint.")
        return
    obj.load_state_dict(checkpoint[key])


def get_adam_optimizer(
    modules: List[Module],
    learning_rate: float,
    adam_eps: float,
    checkpoint: Checkpoint = None,
    key: str = "optimizer",
) -> Optimizer:
    """!
    Create an Adam optimizer and try to load its internal states from the checkpoint.
    @param modules: the modules whose parameters must be optimized
    @param learning_rate: the learning rate
    @param adam_eps: the epsilon parameter of the Adam optimizer
    @param checkpoint: the checkpoint (None if the optimizer must only be created but not loaded)
    @param key: the checkpoint key where the optimizer states are stored
    @return the loaded Adam optimizer
    """

    # Collect the parameters.
    params = []
    for module in modules:
        params += list(module.parameters())

    # Create the optimizer and load its internal states.
    optimizer = optim.Adam(params, lr=learning_rate, eps=adam_eps)
    safe_load_state_dict(optimizer, checkpoint, key)
    return optimizer


def get_sgd_optimizer(
    modules: List[Module],
    learning_rate: float,
    checkpoint: Checkpoint = None,
    key: str = "optimizer",
) -> Optimizer:
    """!
    Create a Stochastic Gradient descent optimizer and try to load its internal states
    from the checkpoint.
    @param modules: the modules whose parameters must be optimized
    @param learning_rate: the learning rate
    @param checkpoint: the checkpoint (None if the optimizer must only be created but not loaded)
    @param key: the checkpoint key where the optimizer states are stored
    @return the loaded Adam optimizer
    """

    # Collect the parameters.
    params = []
    for module in modules:
        params += list(module.parameters())

    # Create the optimizer and load its internal states.
    optimizer = optim.SGD(params, lr=learning_rate)
    safe_load_state_dict(optimizer, checkpoint, key)
    return optimizer
