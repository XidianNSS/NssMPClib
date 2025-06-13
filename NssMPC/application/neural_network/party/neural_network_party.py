#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
import torch.utils.data

from NssMPC.application.neural_network.utils.converter import gen_mat_beaver, image2tensor
from NssMPC.common.utils import bytes_convert
from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.crypto.aux_parameter import MatmulTriples
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *
from NssMPC.secure_model.mpc_party import SemiHonest3PCParty, HonestMajorityParty, SemiHonestCS
from NssMPC.secure_model.utils.param_provider import ParamProvider


class NeuralNetworkCS(SemiHonestCS):
    """
    A neural network class based on a client-server architecture, inherited from SemiHonestCS. It`s main
    function is to support secure multi-party computation of neural network
    models for training and inference, ensuring that calculations are performed in an environment that protects data
    privacy.
    """

    def __init__(self, type):
        """
        Use :meth:`~NssMPC.secure_model.mpc_party.semi_honest.SemiHonestCS.set_multiplication_provider()` to provide support for multiplication operations.
        Use :meth:`~NssMPC.secure_model.mpc_party.semi_honest.SemiHonestCS.set_comparison_provider()` to provide support for comparison operations.
        Use :meth:`~NssMPC.secure_model.mpc_party.semi_honest.SemiHonestCS.set_nonlinear_operation_provider()` to provide support for nonlinear operations (activation functions).

        :param type: Whether the current instance is server-side or client-side
        :type type: str
        """
        super(NeuralNetworkCS, self).__init__(type)
        self.set_multiplication_provider()
        self.set_comparison_provider()
        self.set_nonlinear_operation_provider()

    def activate_by_gelu(self):
        """
        Use the GeLU activation function.

        Use :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider` to add the ability to provide parameters to the Party.
        """
        from NssMPC.crypto.aux_parameter import GeLUKey
        self.append_provider(ParamProvider(param_type=GeLUKey))

    def dummy_model(self, inputs):
        """
        Simulate the model. Generate in advance the parameters needed like the triples for the matrix multiplication.

        TODO: It will need to be modified to a trusted third party in subsequent steps

        First, input processing is performed to determine whether ``self`` is the model provider or the data provider:
            * If it is server:
                If the debug level to 2, not the server from the client receives a virtual input, the called function :func:`~NssMPC.application.neural_network.utils.converter.gen_mat_beaver` can generate the corresponding matrix triad. The resulting triplet is sent back to the client.

            * If it is client:
                First determine the type of input data, and then create a new tensor with the same shape as the given tensor and all elements of 1. When debug level is not 2, virtual input is sent to the server and parameters of the matrix triplet are received.

        :param inputs: the input to the model provider is the class of the model, while the input to the data provider is the data to be inferred.
        :type inputs: torch.Tensor or str or torch.utils.data.DataLoader
        :returns: The amount of input data
        :rtype: torch.tensor
        """
        num = 1
        self.wait()
        if self.type == 'server':
            # 模型提供方
            num = self.receive().item()

            if DEBUG_LEVEL != 2:
                dummy_input = self.receive()
                mat_beaver_lists = gen_mat_beaver(dummy_input=dummy_input.to('cpu'), model=inputs, num_of_triples=num)
                self.providers[MatmulTriples.__name__].param = mat_beaver_lists.pop()
                self.send(mat_beaver_lists.pop())

        if self.type == 'client':
            # 数据提供方
            if isinstance(inputs, torch.Tensor):
                dummy_input = torch.ones_like(inputs)
            elif isinstance(inputs, str):
                dummy_input = torch.ones_like(image2tensor(inputs))
            elif isinstance(inputs, torch.utils.data.DataLoader):
                num = len(inputs.dataset) // inputs.batch_size
                dummy_input = torch.ones_like(enumerate(inputs).__next__()[1][0])
            else:
                raise TypeError("unsupported data type:", type(inputs))
            self.send(torch.tensor(num))

            if DEBUG_LEVEL != 2:
                self.send(dummy_input)
                self.providers[MatmulTriples.__name__].param = self.receive()

        self.providers[MatmulTriples.__name__].load_mat_beaver()
        return num

    def inference(self, net, input_shares):
        """
        Used for neural network inference in encrypted environments.

        First name the processor of the data, then call the neural network model ``net`` and unpack ``input_shares``
        as input. If the current object is a server: the result of the calculation is sent to the client. If the
        current object is a client: after receiving the sent data from the server,
        use the :meth:`~NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing
        .ArithmeticSecretSharing.restore_from_shares` method to restore the result of the calculation, and finally
        transfer the result to the real field.

        :param net: the neural network model
        :type net: torch.nn.Module
        :param input_shares: The shared input data.
        :type input_shares: ArithmeticSecretSharing
        :returns: The data owner(client) output the inference result.
        """
        for data in input_shares:
            data.party = self

        output = net(*input_shares)
        if self.type == 'server':
            self.send(output)
        if self.type == 'client':
            output_other = self.receive()
            output = ArithmeticSecretSharing.restore_from_shares(output_other, output)
            return output.convert_to_real_field()


class NeuralNetwork3PC(SemiHonest3PCParty):
    """
    Used to perform inference on neural networks in a Three Party Computation (3PC) environment.
    """

    def __init__(self, id):
        """
        Initializes an instance of PartyMPC.

        :param id: This participant's ID.
        :type id: int
        """
        super().__init__(id)


class HonestMajorityNeuralNetWork3PC(HonestMajorityParty):
    def __init__(self, party_id):
        """
        Calling the Constructor of the Parent Class to initialize HonestMajorityNeuralNetWork3PC.

        :param party_id: Participant Identity
        :type party_id: int
        """
        super().__init__(party_id)

    def dummy_model(self, *args):
        """
        Process the input data and generate the necessary shared data.

        If it is a **client**:
            After the synchronization wait, first determine the type of ``test_loader`` and create a new tensor with the same shape as the given tensor, and all elements are 1. Then send the batch quantity num to the other party, if DEBUG_LEVEL not 2, call :func:`~NssMPC.application.neural_network.utils.converter.gen_mat_beaver` to generate beaver triples, Send it to another party.

        If it is a **server**:
            The batch number and matrix triples from the first participant are received and stored.

        :return: batch number
        :rtype: int
        """
        self.wait()
        num = 1
        if self.party_id == 0:
            # 数据提供方
            net, test_loader = args
            if isinstance(test_loader, torch.Tensor):
                dummy_input = torch.ones_like(test_loader)
            elif isinstance(test_loader, str):
                dummy_input = torch.ones_like(image2tensor(test_loader))
            elif isinstance(test_loader, torch.utils.data.DataLoader):
                num = len(test_loader.dataset) // test_loader.batch_size
                dummy_input = torch.ones_like(enumerate(test_loader).__next__()[1][0])
            else:
                raise TypeError("unsupported data type:", type(test_loader))

            self.send(1, torch.tensor(num))
            self.send(2, torch.tensor(num))

            if DEBUG_LEVEL != 2:
                mat_beaver_lists = gen_mat_beaver(dummy_input=dummy_input.to('cpu'), model=net, num_of_triples=num,
                                                  num_of_party=3)
                local_t = mat_beaver_lists.pop()

                self.providers[RssMatmulTriples.__name__].param = local_t
                self.send(2, mat_beaver_lists.pop())
                self.send(1, mat_beaver_lists.pop())

        else:
            num = self.receive(0).item()
            if DEBUG_LEVEL != 2:
                local_t = self.receive(0)

                self.providers[RssMatmulTriples.__name__].param = local_t

        self.providers[RssMatmulTriples.__name__].load_mat_beaver()
        return num

    def inference(self, net, input_shares):
        start_send_round = self.communicator.comm_rounds['send']
        start_recv_round = self.communicator.comm_rounds['recv']
        start_send = self.communicator.comm_bytes['send']
        start_recv = self.communicator.comm_bytes['recv']

        output = net(input_shares)

        end_send_round = self.communicator.comm_rounds['send']
        end_recv_round = self.communicator.comm_rounds['recv']
        end_send = self.communicator.comm_bytes['send']
        end_recv = self.communicator.comm_bytes['recv']
        send_round = end_send_round - start_send_round
        recv_round = end_recv_round - start_recv_round
        send = bytes_convert(end_send - start_send)
        recv = bytes_convert(end_recv - start_recv)
        print(f"Communication costs:\n\tsend rounds: {send_round}\t\t"
              f"send bytes: {send}.")
        print(f"\trecv rounds: {recv_round}\t\t"
              f"recv bytes: {recv}.")

        output = output.restore()
        return output.convert_to_real_field()

    def wait(self):
        """
        Ensure synchronisation of communications between the two parties.

        Each party sends a message to the other two first to ensure that the communication is synchronized.
        """
        self.send((self.party_id + 1) % 3, 0)
        self.send((self.party_id + 2) % 3, 0)
        self.receive((self.party_id + 1) % 3)
        self.receive((self.party_id + 2) % 3)
