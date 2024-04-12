import inspect
import os
from collections import OrderedDict

import cv2 as cv
import torchvision.transforms as transforms
from torchvision.transforms import transforms

from application.neural_network.functional.beaver_for_layers import *
from application.neural_network.model.model_compiler import sec_format
from common.tensor.ring_tensor import RingTensor
from config.base_configs import DEVICE
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor


def share_and_save_model(model, save_path, num_of_party=2):
    """
    For the model holder.
    Load the parameters of the plaintext model and share them into n shares,
    which are the parameters of the ciphertext model. Besides, save these shares.

    Args:
        model: the class of plaintext model
        save_path: path to save the shared ciphertext weights
        num_of_party: the number of parties
    """
    param_list = share_model(model, num_of_party)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(num_of_party):
        torch.save(param_list[i], save_path + f'/{model.__class__.__name__}_params_{str(i)}.pkl')


def load_secure_model_from_file(net, path, party):
    """
    For the computation party
    Args:
        net: the class of plaintext model.
        path: Path to the ciphertext weights held by this party.
        party:

    Returns:
        Cipher algorithm model loaded with weights
    """
    party.wait()
    param_dict = torch.load(path + f'/{net.__class__.__name__}_params_{str(party.party_id)}.pkl', map_location=DEVICE)
    return load_secure_model(net, param_dict)


def load_secure_model(net, param_dict):
    """
       For the computation party
       Args:
           net: the class of plaintext model
           param_dict: the parameters of the ciphertext model

       Returns:
           Cipher algorithm model loaded with weights
       """
    model_file_path = inspect.getsourcefile(net.__class__)
    exec(sec_format(model_file_path), locals())
    net = locals()[net.__class__.__name__]()
    net.train(False)
    for name, param in net.named_parameters():
        if name in param_dict:
            param.data = param_dict[name]
    return net


def share_model(model, num_of_party=2):
    """
     For the model holder.
     Load the parameters of the plaintext model and share them into n shares,
     which are the parameters of the ciphertext model. Besides, save these shares.

     Args:
         model: the class of plaintext model
         num_of_party: the number of parties
     """

    def modify_bn_layers(module):
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

    modify_bn_layers(model)

    param_dict_list = [OrderedDict() for _ in range(num_of_party)]
    for name, param in model.state_dict().items():
        ring_param = RingTensor.convert_to_ring(param)
        shares = ArithmeticSharedRingTensor.share(ring_param, num_of_party)
        for param_dict, share in zip(param_dict_list, shares):
            param_dict[name] = share.tensor
    return param_dict_list


def share_data(*inputs, num_of_party=2):
    """
    For the data holder.
    share the input data into n shares.
    Args:
        inputs: the input data
        num_of_party: the number of parties
    """
    input_shares = []
    for i in range(num_of_party):
        input_shares.append([])
    for input_info in inputs:
        if isinstance(input_info, str):
            # read data from file
            input_info = image2tensor(input_info)
        elif isinstance(input_info, torch.Tensor):
            input_info = input_info
        else:
            raise TypeError("unsupported data type:", type(input_info))
        input_list = ArithmeticSharedRingTensor.share(RingTensor.convert_to_ring(input_info), num_of_party)
        for i in range(num_of_party):
            input_shares[i].append(input_list[i])
    return input_shares


def image2tensor(image_path):
    """
    Convert images to tensors

    Args:
        image_path: the path of the image

    Returns:
        tensor: the tensor representing the image
    """
    if not image_path.split('.')[-1] in ('jpg', 'png', 'bmp'):
        raise TypeError("unsupported file type:", image_path.split('.')[-1])
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(DEVICE)


def gen_mat_beaver(dummy_input, model, num_of_triples, num_of_party=2):
    """
    For the model holder.
    Generate the beavers for the matrix calculation of the ciphertext model.

    Args:
        dummy_input: analogue inputs of the same size as the neural network inputs
        model: the class of plaintext model
        num_of_triples
        num_of_party
    """
    mat_beaver_lists = [[] for _ in range(num_of_party)]

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
