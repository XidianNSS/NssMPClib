import warnings

import torch

from nssmpc import Party2PC, Party3PC, HONEST_MAJORITY
from nssmpc.infra.mpc import Party, PartyCtx
from nssmpc.protocols.honest_majority_3pc.multiplication import RssMatmulTriples
from nssmpc.protocols.semi_honest_2pc.multiplication import MatmulTriples


def gen_mat_beaver(
        dummy_input: torch.Tensor,
        model: torch.nn.Module,
        num_of_triples: int,
        party: Party = None,
        save: bool = False
):
    """Generate Beaver triples for specific layers by registering a forward hook.

    Args:
        dummy_input (torch.Tensor): Analogue inputs of the same size as the neural communication inputs.
        model (torch.nn.Module): The class of plaintext model.
        num_of_triples (int): The number of Beaver triples to be generated.
        num_of_party (int): The number of parties involved in the computation. Defaults to 2.
        party: The party instance. If None, uses the current context specific party.
        save: If True, triples are saved; if False, triples are returned.

    Returns:
        If save is False: list: The list of Beaver triples for each party.
        If save is True: None (triples are saved).
        If party type doesn't need triples: None.

    Examples:
        >>> # generate triples and return them without saving
        >>> beavers = gen_mat_beaver(input, model, 100, save=False)
        >>> # generate triples and save them
        >>> gen_mat_beaver(input, model, 100, party, save=True)
    """
    generator = None
    num_of_party = None
    mat_beaver_lists = None

    if party is None:
        party = PartyCtx.get()

    if isinstance(party, Party2PC):
        num_of_party = 2
        if save:
            generator = MatmulTriples.gen_and_save
        else:
            generator = MatmulTriples.gen
    elif isinstance(party, Party3PC) and party.thread_model_cfg == HONEST_MAJORITY:
        num_of_party = 3
        if save:
            generator = RssMatmulTriples.gen_and_save
        else:
            generator = RssMatmulTriples.gen
    else:
        warnings.warn("This party does not need to generate beaver triples.")

    if generator is None:
        return None

    if not save:
        mat_beaver_lists = [[] for _ in range(num_of_party)]

    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            result = _generate_beaver_for_conv(
                input[0], module.weight, module.padding[0], module.stride[0],
                num_of_triples, generator
            )
        elif isinstance(module, torch.nn.modules.linear.Linear):
            result = _generate_beaver_for_linear(
                input[0], module.weight, num_of_triples, generator
            )
        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            result = _generate_beaver_for_avg_pooling(
                input[0], module.kernel_size, module.padding, module.stride,
                num_of_triples, generator
            )
        elif isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            result = _generate_beaver_for_adaptive_avg_pooling(
                input[0], module.output_size, num_of_triples, generator
            )
        else:
            return

        if not save:
            for i in range(num_of_party):
                mat_beaver_lists[i].append(result[i])

    def register_hooks(module, hook):
        hooks = []
        for child in module.children():
            hooks.append(child.register_forward_hook(hook))
            hooks += register_hooks(child, hook)
        return hooks

    hooks = register_hooks(model, hook_fn)
    model(dummy_input)

    for hook in hooks:
        hook.remove()

    if not save:
        return mat_beaver_lists


def _generate_beaver_for_conv(x, kernel, padding, stride, num_of_triples, generator):
    """Internal function to generate matrix Beaver triples for convolutional layers.

    Args:
        x (torch.Tensor or RingTensor): The input tensor, used to determine shapes.
        kernel (torch.Tensor): The convolutional kernel tensor.
        padding (int): The amount of implicit padding on both sides.
        stride (int): The stride of the convolving kernel.
        num_of_triples (int): The number of triples to generate.
        generator: The generator function to use.

    Returns:
        The result from the generator function.
    """
    n, c, h, w = x.shape
    f, _, k, _ = kernel.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    im2col_output_shape = torch.zeros([n, h_out * w_out, c * k * k]).shape
    reshaped_kernel_size = torch.zeros([1, c * k * k, f]).shape

    return generator(num_of_triples, im2col_output_shape, reshaped_kernel_size)


def _generate_beaver_for_linear(x, weight, num_of_triples, generator):
    """Internal function to generate matrix Beaver triples for linear layers.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor of the linear layer.
        num_of_triples (int): The number of triples to generate.
        generator: The generator function to use.

    Returns:
        The result from the generator function.
    """
    weight = weight.T
    return generator(num_of_triples, x.shape, weight.shape)


def _generate_beaver_for_avg_pooling(x, kernel_shape, padding, stride, num_of_triples, generator):
    """Internal function to generate matrix Beaver triples for average pooling layers.

    Args:
        x (torch.Tensor): The input tensor.
        kernel_shape (int): The size of the pooling window (kernel size).
        padding (int): The amount of implicit padding on both sides.
        stride (int): The stride of the pooling window.
        num_of_triples (int): The number of triples to generate.
        generator: The generator function to use.

    Returns:
        The result from the generator function.
    """
    n, c, h, w = x.shape
    h_out = (h + 2 * padding - kernel_shape) // stride + 1
    w_out = (w + 2 * padding - kernel_shape) // stride + 1
    shapes = torch.zeros([n, c, h_out * w_out, kernel_shape * kernel_shape]).shape
    second_shape = torch.zeros([1, 1, shapes[3], 1]).shape

    return generator(num_of_triples, shapes, second_shape)


def _generate_beaver_for_adaptive_avg_pooling(x, output_shape, num_of_triples, generator):
    """Internal function to generate matrix beaver triples for adaptive average pooling layers.

    Args:
        x (torch.Tensor): The input tensor.
        output_shape (tuple): The shape of the output tensor.
        num_of_triples (int): The number of triples.
        generator: The generator function to use.

    Returns:
        The result from the generator function.
    """
    input_shape = torch.tensor(x.shape[2:])
    output_shape = torch.tensor(output_shape)

    stride = torch.floor(input_shape / output_shape).to(torch.int64)
    kernel_size = input_shape - (output_shape - 1) * stride

    kernel_size_list = kernel_size.tolist()
    stride_list = stride.tolist()

    return _generate_beaver_for_avg_pooling(
        x, kernel_shape=kernel_size_list[0], padding=0, stride=stride_list[0],
        num_of_triples=num_of_triples, generator=generator
    )
