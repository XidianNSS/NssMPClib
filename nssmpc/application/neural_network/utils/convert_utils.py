#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import inspect
from collections import OrderedDict

import torch
import torch.nn.functional as F

from nssmpc import Party2PC, Party3PC
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing, ReplicatedSecretSharing
from nssmpc.runtime.context import PartyCtx
from nssmpc.application.neural_network.utils.model_converter import _sec_format
from nssmpc.config.configs import DEVICE
from nssmpc.infra.tensor import RingTensor


def share_model_param(*, model=None, src_id=None):
    """Load the parameters of the plaintext model and share them into n shares.

    This function converts the model parameters to RingTensors and then shares them
    using either Arithmetic Secret Sharing (type 22) or Replicated Secret Sharing (type 32).
    It also handles Batch Normalization layer adjustments.

    Args:
        model (torch.nn.Module): The class of plaintext model.
        src_id (int): The source party ID from which to receive the shared parameters.

    Returns:
        OrderedDict: The shared parameters of the model.

    Examples:
        #For model owner (e.g., Party 0)
        >>> shared_param_dict = share_model_param(model)
        #For other parties
        >>> shared_param_dict = share_model_param(src_id=0)
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

    assert (model is not None) ^ (src_id is not None), "Exactly one of model or src_id must be provided."

    party = PartyCtx.get()
    if model is None and src_id is None:
        raise ValueError("Either `model` or `src_id` must be provided.")
    if isinstance(party, Party2PC):
        share_type = AdditiveSecretSharing
        num_of_party = 2
    elif isinstance(party, Party3PC):
        share_type = ReplicatedSecretSharing
        num_of_party = 3
    else:
        raise RuntimeError("Unsupported party type for SecretTensor.")
    if model is not None:
        modify_bn_layers(model)
        param_dict_list = [OrderedDict() for _ in range(num_of_party)]

        for name, param in model.state_dict().items():
            shares = share_type.share(RingTensor.convert_to_ring(param))
            for param_dict, share in zip(param_dict_list, shares):
                param_dict[name] = share.item.tensor

        if isinstance(party, Party2PC):
            party.send(param_dict_list[1])
        elif isinstance(party, Party3PC):
            party.send((party.party_id + 1) % 3, param_dict_list[1])
            party.send((party.party_id + 2) % 3, param_dict_list[2])
        return param_dict_list[0]

    elif src_id is not None:
        if isinstance(party, Party2PC):
            return party.recv()
        elif isinstance(party, Party3PC):
            return party.recv(src_id)


def convert_model(model_class: type):
    """Convert a plaintext model class into a secure model class.

    This function dynamically executes the model definition code (processed by `sec_format`)
    to instantiate the secure model.

    Args:
        model_class (type): The class of plaintext model.

    Returns:
        type: The converted secure model class.

    Examples:
        >>> secure_model = convert_model(Net)
    """
    model_file_path = inspect.getsourcefile(model_class)
    exec(_sec_format(model_file_path), locals())
    secure_model = locals()[model_class.__name__]
    return secure_model


def load_shared_param(model_instance, param_dict):
    """Load an encryption algorithm model and convert its parameters from ciphertext format.

    This function dynamically executes the model definition code (processed by `sec_format`)
    to instantiate the secure model and loads the provided parameters.

    Args:
        model_instance (torch.nn.Module): The class of plaintext model.
        param_dict (dict): The parameters of the ciphertext model.

    Returns:
        torch.nn.Module: Cipher algorithm model loaded with weights.

    Examples:
        >>> model = load_shared_param(Net(), param_dict)
    """
    for name, param in model_instance.named_parameters():
        if name in param_dict:
            param.data = param_dict[name].to(DEVICE)
    return model_instance


def SharedDataLoader(*, data_loader: torch.utils.data.DataLoader = None, src_id: int = None):
    """Convert a plaintext data loader into a secure data loader.

    This function shares each batch of data in the data loader using secret sharing.

    Args:
        data_loader (torch.utils.data.DataLoader): The plaintext data loader.
        src_id (int): The source party ID from which to receive the shared data.

    Returns:
        torch.utils.data.DataLoader: The converted secure data loader.

    Examples:
        #For data owner (e.g., Party 0)
        >>> secure_data_loader = SharedDataLoader(data_loader=my_data_loader)
        #For other parties
        >>> secure_data_loader = SharedDataLoader(src_id=0)
    """
    assert (data_loader is not None) ^ (src_id is not None), "Exactly one of data_loader or src_id must be provided."

    party = PartyCtx.get()

    if isinstance(party, Party2PC):
        share_type = AdditiveSecretSharing
    elif isinstance(party, Party3PC):
        share_type = ReplicatedSecretSharing
    else:
        raise RuntimeError("Unsupported party type for SecretTensor.")

    if data_loader is not None:
        for batch in data_loader:
            label = None
            if isinstance(batch, (list, tuple)):
                batch, label = batch
            shared_batches = share_type.share(RingTensor.convert_to_ring(batch))
            if isinstance(party, Party2PC):
                party.send(shared_batches[1])
            elif isinstance(party, Party3PC):
                party.send((party.party_id + 1) % 3, shared_batches[1])
                party.send((party.party_id + 2) % 3, shared_batches[2])
            if label is not None:
                yield shared_batches[0].to(DEVICE), label.to(DEVICE)
            else:
                yield shared_batches[0].to(DEVICE)
        if isinstance(party, Party2PC):
            party.send("END")
        elif isinstance(party, Party3PC):
            party.send((party.party_id + 1) % 3, "END")
            party.send((party.party_id + 2) % 3, "END")
    elif src_id is not None:
        while True:
            if isinstance(party, Party2PC):
                received = party.recv()
            elif isinstance(party, Party3PC):
                received = party.recv(src_id)
            else:
                raise RuntimeError("Unsupported party type for SecretTensor.")
            if isinstance(received, str):
                break
            yield received.to(DEVICE)
    return


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
