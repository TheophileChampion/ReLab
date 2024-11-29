import torch

import benchmarks
from benchmarks import agents

def describe(parameters, n_copies=1, memory_unit="GB"):
    """
    Describe the parameters passed as arguments.
    :param parameters: the parameters to describe
    :param n_copies: the number of copies the agent has
    :param memory_unit: the unit to use describe the model size (B, KB, MB or GB)
    """

    # The size of one parameter in bits for each data type.
    dtype_sizes = {
        torch.complex128: 128, torch.cdouble: 128,
        torch.float64: 64, torch.double: 64, torch.complex64: 64, torch.cfloat: 64, torch.int64: 64, torch.long: 64,
        torch.float32: 32, torch.float: 32, torch.int32: 32, torch.int: 32,
        torch.float16: 16, torch.half: 16, torch.bfloat16: 16, torch.int16: 16, torch.short: 16,
        torch.uint8: 8, torch.int8: 8,
    }

    # The number of bytes for each memory unit.
    memory_units = {
        "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9,
    }

    # Count the number of parameters, keep track of their types, and memory usage.
    dtypes = []
    memory_usage = 0
    total_n_params = 0
    for params in parameters:
        dtype = params.dtype
        dtypes.append(dtype)
        n_params = n_copies * params.numel()
        total_n_params += n_params
        memory_usage += n_params * dtype_sizes[dtype] / (8 * memory_units[memory_unit])

    # Display the parameter summary on the standard output.
    if all([dtype == dtypes[0] for dtype in dtypes]) is True:
        dtypes = dtypes[0]
    print(f"Parameter type: {dtypes}.")
    print(f"Number of parameters: {total_n_params}.")
    print(f"Parameters memory size: {memory_usage:0.3f} {memory_unit}.")


def describe_params(agent_name, env_name, seed):
    """
    Describe the agent's parameters.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent_name, env_name, seed)

    # Create the requested agent.
    agent = agents.make(agent_name, training=True)

    # Describe the agent parameters.
    describe(agent.value_net.parameters(), n_copies=2)

if __name__ == "__main__":

    # Describe the agent's parameters.
    describe_params(agent_name="DuelingDDQN", env_name="ALE/Pong-v5", seed=0)
