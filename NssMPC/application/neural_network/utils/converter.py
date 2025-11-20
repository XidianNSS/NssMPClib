#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import inspect
import os
from collections import OrderedDict

import torch

from NssMPC.application.neural_network.utils.model_compiler import sec_format
from NssMPC.infra.tensor import RingTensor
from NssMPC.config.configs import DEVICE


def share_model(model, share_type=22):
    """Load the parameters of the plaintext model and share them into n shares.

    This function converts the model parameters to RingTensors and then shares them
    using either Arithmetic Secret Sharing (type 22) or Replicated Secret Sharing (type 32).
    It also handles Batch Normalization layer adjustments.

    Args:
        model (torch.nn.Module): The class of plaintext model.
        share_type (int): An integer indicating the shared type (22 or 32). Defaults to 22.

    Returns:
        list: A list of dictionaries, where each dictionary contains the shared parameters for a participant.

    Examples:
        >>> param_shares = share_model(model, share_type=22)
    """

    # TODO: Transformer模型可能存在无法加载明文参数的情况，此时是否修改为再传一个明文参数的state_dict
    def modify_bn_layers(module):
        """Modify the parameters of the Batch Normalization layer.

        Args:
            module (torch.nn.Module): The class of plaintext model.

        Examples:
            >>> modify_bn_layers(model)
        """
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            gamma = module.weight.data
            beta = module.bias.data
            running_mean = module.running_mean
            running_var = module.running_var + module.eps
            module.weight.data = gamma / running_var.sqrt()
            module.bias.data = beta - (gamma * running_mean) / running_var.sqrt()
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None
        else:
            for child in module.children():
                modify_bn_layers(child)

    assert share_type in [22, 32]
    num_of_party = share_type // 10

    if share_type == 22:
        from NssMPC.primitives.secret_sharing import AdditiveSecretSharing as ShareType
    elif share_type == 32:
        from NssMPC.primitives import ReplicatedSecretSharing as ShareType
    modify_bn_layers(model)

    param_dict_list = [OrderedDict() for _ in range(num_of_party)]
    for name, param in model.state_dict().items():
        ring_param = RingTensor.convert_to_ring(param)
        shares = ShareType.share(ring_param)
        for param_dict, share in zip(param_dict_list, shares):
            param_dict[name] = share.item.tensor
    return param_dict_list


def share_and_save_model(model, save_path, num_of_party=2):
    """Securely split the given plaintext model and save the shared model parameters.

    Args:
        model (torch.nn.Module): The class of plaintext model.
        save_path (str): Path to save the shared ciphertext weights.
        num_of_party (int): The number of parties. Defaults to 2.

    Examples:
        >>> share_and_save_model(model, './params', num_of_party=2)
    """
    param_list = share_model(model, num_of_party)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(num_of_party):
        torch.save(param_list[i], save_path + f'/{model.__class__.__name__}_params_{str(i)}.pkl')


def load_model_from_file(net, path, party):
    """Load the model parameters from a file for the computation party.

    Args:
        net (torch.nn.Module): The class of plaintext model.
        path (str): Path to the ciphertext weights held by this party.
        party: The participant object.

    Returns:
        torch.nn.Module: Cipher algorithm model loaded with weights.

    Examples:
        >>> model = load_model_from_file(Net(), './params', party)
    """
    party.wait()
    param_dict = torch.load(path + f'/{net.__class__.__name__}_params_{str(party.party_id)}.pkl', map_location=DEVICE)
    return load_model(net, param_dict)


def load_model(net, param_dict):
    """Load an encryption algorithm model and convert its parameters from ciphertext format.

    This function dynamically executes the model definition code (processed by `sec_format`)
    to instantiate the secure model and loads the provided parameters.

    Args:
        net (torch.nn.Module): The class of plaintext model.
        param_dict (dict): The parameters of the ciphertext model.

    Returns:
        torch.nn.Module: Cipher algorithm model loaded with weights.

    Examples:
        >>> model = load_model(Net(), param_dict)
    """
    model_file_path = inspect.getsourcefile(net.__class__)
    exec(sec_format(model_file_path), locals())
    net = locals()[net.__class__.__name__]()  # TODO: 部分模型创建时需要传入参数
    net.train(False)
    for name, param in net.named_parameters():
        if name in param_dict:
            param.data = param_dict[name].to(DEVICE)
    return net


def share_data(*inputs, share_type=22):
    """Perform secret sharing on input data for the data owner.

    Args:
        *inputs: The input data (torch.Tensor) or paths to images (str).
        share_type (int): The sharing type (22 or 32). Defaults to 22.

    Returns:
        list: The shared data list.

    Raises:
        TypeError: If an unsupported data type is provided as input.

    Examples:
        >>> shares = share_data(input_tensor, share_type=22)
    """
    assert share_type in [22, 32]
    num_of_party = share_type // 10

    if share_type == 22:
        from NssMPC.primitives import AdditiveSecretSharing as ShareType
    elif share_type == 32:
        from NssMPC.primitives import ReplicatedSecretSharing as ShareType

    input_shares = [[] for _ in range(num_of_party)]
    for input_info in inputs:
        if isinstance(input_info, str):
            # read data from file
            input_info = image2tensor(input_info)
        elif isinstance(input_info, torch.Tensor):
            input_info = input_info
        else:
            raise TypeError("unsupported data type:", type(input_info))
        input_list = ShareType.share(RingTensor.convert_to_ring(input_info))
        for i in range(num_of_party):
            input_shares[i].append(input_list[i])
    return input_shares


def image2tensor(image_path):
    """Check whether the given image_path file path ends with a supported image file type.

    Args:
        image_path (str): The path of the image.

    Returns:
        torch.Tensor: The tensor representing the image.

    Raises:
        TypeError: If the extension is not among the supported file types.

    Examples:
        >>> tensor = image2tensor('image.jpg')
    """
    if not image_path.split('.')[-1] in ('jpg', 'png', 'bmp'):
        raise TypeError("unsupported file type:", image_path.split('.')[-1])
    # TODO 图片转tensor用torch实现
    # image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # transform = transforms.ToTensor()
    # image = transform(image)
    # image = image.unsqueeze(0)
    # return image.to(DEVICE)


def gen_mat_beaver(dummy_input, model, num_of_triples, num_of_party=2):
    """Generate Beaver triples for specific layers by registering a forward hook.

    Args:
        dummy_input (torch.Tensor): Analogue inputs of the same size as the neural communication inputs.
        model (torch.nn.Module): The class of plaintext model.
        num_of_triples (int): The number of Beaver triples to be generated.
        num_of_party (int): The number of parties involved in the computation. Defaults to 2.

    Returns:
        list: The list of Beaver triples for each party.

    Examples:
        >>> beavers = gen_mat_beaver(input, model, 100)
    """
    mat_beaver_lists = [[] for _ in range(num_of_party)]
    from NssMPC.application.neural_network.functional.beaver_for_layers import beaver_for_adaptive_avg_pooling, \
        beaver_for_avg_pooling, beaver_for_linear, beaver_for_conv

    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            mat_beavers = beaver_for_conv(input[0], module.weight, module.padding[0], module.stride[0], num_of_triples)
        elif isinstance(module, torch.nn.modules.linear.Linear):
            mat_beavers = beaver_for_linear(input[0], module.weight, num_of_triples)
        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            mat_beavers = beaver_for_avg_pooling(input[0], module.kernel_size, module.padding, module.stride,
                                                 num_of_triples)
        elif isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            mat_beavers = beaver_for_adaptive_avg_pooling(input[0], module.output_size, num_of_triples)
        else:
            return
        for i in range(num_of_party):
            mat_beaver_lists[i].append(mat_beavers[i])

    def register_hooks(module, hook):
        """Register forward hooks recursively.

        Args:
            module (torch.nn.Module): The current layer.
            hook (callable): Forward hook function.

        Returns:
            list: List of registered hooks.

        Examples:
            >>> hooks = register_hooks(model, hook_fn)
        """
        hooks = []
        for child in module.children():
            hooks.append(child.register_forward_hook(hook))
            hooks += register_hooks(child, hook)
        return hooks

    hooks = register_hooks(model, hook_fn)
    model(dummy_input)

    for hook in hooks:
        hook.remove()

    return mat_beaver_lists


def embedding_preparation(inputs, each_embedding_size):
    """Convert input data into one-hot encoding for the embedding layer.

    Args:
        inputs (torch.Tensor or list or tuple): The input data to be transformed.
        each_embedding_size (int or list or tuple): The size of each embedding vector.

    Returns:
        list: The one-hot encoding of the input data.

    Raises:
        TypeError: If the inputs is not among the supported data types.

    Examples:
        >>> one_hot = embedding_preparation(inputs, 10)
    """
    if isinstance(inputs, torch.Tensor):
        return F.one_hot(inputs.to(torch.int64), num_classes=each_embedding_size).to(inputs.dtype) * 1.0
    elif isinstance(inputs, (list, tuple)):
        assert isinstance(each_embedding_size, (list, tuple))
        outputs = []
        for input_data, embedding_size in zip(inputs, each_embedding_size):
            outputs.append(
                F.one_hot(input_data.to(torch.int64), num_classes=embedding_size).to(input_data.dtype) * 1.0)
        return outputs
    else:
        raise TypeError("unsupported data type:", type(inputs))
