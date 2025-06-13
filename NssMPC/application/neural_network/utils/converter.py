#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import inspect
import os
from collections import OrderedDict

from NssMPC.application.neural_network.utils.model_compiler import sec_format
from NssMPC.common.ring.ring_tensor import *
from NssMPC.config.configs import DEVICE


def share_model(model, share_type=22):
    """
    For the model holder.
    Load the parameters of the plaintext model and share them into n shares,
    which are the parameters of the ciphertext model.

    then, Make sure that the ``share_type`` is set to 22 or 32,
    and calculate the number of participants (2 or 3). Based on the value of share_type, import the corresponding sharing method:
        * 22 represents Arithmetic Secret Sharing.
        * 32 represents Replicated Secret Sharing.

    Then, call the internal function to ensure that the BatchNorm layer parameters in the model have been properly
    adjusted. After initializing a list of ``num_of_party`` element parameter dictionaries, where each dictionary is
    used to store the shared parameters for the corresponding participant, traversing the model's parameter
    dictionary.The parameters are shared after each parameter is converted to the ring, and the shared values are
    stored in a dictionary.

    :param model: the class of plaintext model
    :type model: torch.nn.Module
    :param share_type: An integer indicating the shared type, which determines the number of participants and the sharing method. Supports 22 and 32.
    :type share_type: int
    """

    # TODO: Transformer模型可能存在无法加载明文参数的情况，此时是否修改为再传一个明文参数的state_dict
    def modify_bn_layers(module):
        """
        Modify the parameters of the Batch Normalization layer to adapt to the subsequent shared process.
        For the BatchNorm layer, extract its ``gamma``, ``beta``, ``running_mean``, and ``running_var``. For each
        submodule, recursively call :func:`modify_bn_layers`.

        :param module: the class of plaintext model
        :type module: torch.nn.Module
        :return: the parameters of the Batch Normalization layer
        :rtype: list
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
        from NssMPC import ArithmeticSecretSharing as ShareType
    elif share_type == 32:
        from NssMPC import ReplicatedSecretSharing as ShareType
    modify_bn_layers(model)

    param_dict_list = [OrderedDict() for _ in range(num_of_party)]
    for name, param in model.state_dict().items():
        ring_param = RingTensor.convert_to_ring(param)
        shares = ShareType.share(ring_param)
        for param_dict, share in zip(param_dict_list, shares):
            param_dict[name] = share.item.tensor
    return param_dict_list


def share_and_save_model(model, save_path, num_of_party=2):
    """
    Securely split the given plaintext model and save the shared model parameters to the specified path.

    First, the :func:`share_model` function is called to secretly share the parameters of the model, check whether the
    specified saving path exists, and create the directory if it does not exist. Finally, the parameters of each
    participant are saved in the specified path

    :param model: the class of plaintext model
    :type model: torch.nn.Module
    :param save_path: path to save the shared ciphertext weights
    :type save_path: str
    :param num_of_party: the number of parties
    :type num_of_party: int
    """
    param_list = share_model(model, num_of_party)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(num_of_party):
        torch.save(param_list[i], save_path + f'/{model.__class__.__name__}_params_{str(i)}.pkl')


def load_model_from_file(net, path, party):
    """
    For the computation party.First load the parameter path, then call the function :func:`load_model` to load the model.

    :param net: the class of plaintext model.
    :type net: torch.nn.Module
    :param path: Path to the ciphertext weights held by this party.
    :type path: str
    :param party: participant
    :type party: str

    Returns:
        Cipher algorithm model loaded with weights
    """
    party.wait()
    param_dict = torch.load(path + f'/{net.__class__.__name__}_params_{str(party.party_id)}.pkl', map_location=DEVICE)
    return load_model(net, param_dict)


def load_model(net, param_dict):
    """
    For the computation party.Load an encryption algorithm model and convert its parameters from ciphertext format to plaintext format.

    After obtaining the source file path of the current model class, execute the code defined by the model using exec.

    .. note::
        *locals()* is passed to *exec*, which makes all local variables available in the current context when the model definition is executed.

    The model is then instantiated, obtaining the name of the model class from the local namespace, and creating an instance of the class. After the model is set to evaluation mode, the parameters in the parameter dictionary are loaded into the model parameters.

    :param net: the class of plaintext model
    :type net: torch.nn.Module
    :param param_dict: the parameters of the ciphertext model
    :type param_dict: dict
    :returns: Cipher algorithm model loaded with weights
    :rtype: torch.nn.Module
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
    """
    Perform secret sharing on input data for the data owner.

    This method supports two different sharing types (22 and 32) and returns a list of shared data.
    It securely splits the provided input data into shares for multiple parties, facilitating secure multi-party computation.

    The function determines the sharing method based on the `share_type` parameter. For `share_type` 22,
    it uses Arithmetic Secret Sharing, while for 32, it uses Replicated Secret Sharing. It then processes each
    input, performing necessary conversions and sharing operations.

    :param inputs: the input data
    :type inputs: torch.Tensor
    :param share_type: the number of parties
    :type share_type: int
    :returns: the shared data list
    :rtype: list
    :raises TypeError: If an unsupported data type is provided as input.
    """
    assert share_type in [22, 32]
    num_of_party = share_type // 10

    if share_type == 22:
        from NssMPC import ArithmeticSecretSharing as ShareType
    elif share_type == 32:
        from NssMPC import ReplicatedSecretSharing as ShareType

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
    """
    Check whether the given image_path file path ends with a supported image file type.

    First press the file path to the point (.) via the ``split('.')`` method. Split into a list and take the last element
    of the list, then check whether the extracted file extension is not in the supported extension tuple ('jpg',
    'png', 'bmp').

    :param image_path: the path of the image
    :type image_path: str
    :returns: the tensor representing the image
    :rtype: torch.Tensor
    :raises TypeError: If the extension is not among the supported file types.
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
    """
    For the model holder. By registering a forward hook, you can monitor specific types of layers and generate corresponding Beaver triples for each layer.

    :param dummy_input: Analogue inputs of the same size as the neural network inputs.
    :type dummy_input: torch.Tensor
    :param model: the class of plaintext model
    :type model: torch.nn.Module
    :param num_of_triples: The number of Beaver triples to be generated.
    :type num_of_triples: int
    :param num_of_party: The number of parties involved in the computation.
    :type num_of_party: int
    :returns: The list of Beaver triples for each party.
    :rtype: list

    - :func:`hook_fn`
        Generate a Beaver triad for each layer.

        This is a callback function that is called when a certain layer of the neural network is executed,
        calling the corresponding function based on the type of the current layer to generate the Beaver triplet,
        and storing the result.

        .. note::
            Hooks are usually temporary and only needed during specific forward or backward propagation
            processes. They are registered and used immediately, then removed to avoid any impact on subsequent operations.


        PARAMETERS:
            * **module** (*torch.nn.Module*): The current layer.
            * **input** (*torch.Tensor*): input data
            * **output** (*torch.Tensor*): output data

        RETURNS:
            generated Beaver triples

        RETURN TYPE:
            list

    - :func:`register_hooks`
        Register forward hooks.

        The register_hooks function recursively registers forward hooks for each submodule

        PARAMETERS:
            * **module** (*torch.nn.Module*): The current layer.
            * **hook** (*list*): Forward hook

        RETURNS:
            List of registered hooks.

        RETURN TYPE:
            list

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
        """
        Register forward hooks.

        The register_hooks function recursively registers forward hooks for each submodule

        :param module: The current layer
        :type module: torch.nn.Module
        :param hook: Forward hook
        :type hook: func
        :return: a function registered a hook.
        :rtype: list
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
    """
    Responsible for converting input data into one-hot encoding, mainly used to prepare the input for the embedding layer.

    First determine the type of inputs:
        For torch.Tensor: First convert the input to the integer type torch.int64, then use the ``F.on_hot`` function to generate a unique thermal encoding

        For list or tuple: first assert that ``each_embedding_size`` is also a list or tuple to make sure the length is the same, then perform the same unique heat encoding process for each input data as above, adding the results to the outputs list.


    :param inputs: the input data to be transformed.
    :type inputs: torch.Tensor
    :param each_embedding_size: the size of each embedding vector.
    :type each_embedding_size: int or list or tuple
    :returns: the one-hot encoding of the input data.
    :rtype: list
    :raises TypeError: If the inputs is not among the supported data types.
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
