#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
import torch.utils.data

from NssMPC.application.neural_network.utils.converter import gen_mat_beaver, image2tensor
from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.infra.mpc.party import Party2PC, Party3PC
from NssMPC.infra.utils.debug_utils import bytes_convert
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing
from NssMPC.protocols.honest_majority_3pc.multiplication import RssMatmulTriples
from NssMPC.protocols.semi_honest_2pc.multiplication import MatmulTriples
from NssMPC.runtime import HONEST_MAJORITY


class PartyNeuralNetwork2PC(Party2PC):
    """
    A neural communication class based on a client-server architecture, inherited from SemiHonestCS.

    It supports secure multi-party computation of neural communication models for training and inference,
    ensuring that calculations are performed in an environment that protects data privacy.
    """

    def dummy_model(self, inputs):
        """
        Simulates the model execution to generate necessary parameters like triples for matrix multiplication.

        This method processes inputs to determine if `self` is the model provider or data provider.
        It handles the generation and exchange of Beaver triples based on the role and debug level.

        Args:
            inputs (torch.Tensor | str | torch.utils.data.DataLoader): The input to the model provider is the class
                of the model, while the input to the data provider is the data to be inferred.

        Returns:
            torch.Tensor: The amount of input data (batch size or number of samples).

        Examples:
            num = party.dummy_model(model)
        """
        num = 1
        self.wait()
        if self.party_id == 0:
            # 模型提供方
            num = self.recv().item()
            MatmulTriples.load_provider(self)
            if DEBUG_LEVEL != 2:
                dummy_input = self.recv()
                mat_beaver_lists = gen_mat_beaver(dummy_input=dummy_input.to('cpu'), model=inputs, num_of_triples=num)
                self.providers[MatmulTriples.__name__].param = mat_beaver_lists.pop()
                self.send(mat_beaver_lists.pop())

        if self.party_id == 1:
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

            MatmulTriples.load_provider(self)

            if DEBUG_LEVEL != 2:
                self.send(dummy_input)
                self.providers[MatmulTriples.__name__].param = self.recv()

        self.providers[MatmulTriples.__name__].load_mat_beaver()
        return num

    def inference(self, net, input_shares):
        """
        Performs neural network inference in an encrypted environment.

        It calls the neural network model `net` with `input_shares`.
        If server: sends calculation result to client.
        If client: receives data from server, restores the result, and converts to real field.

        Args:
            net (torch.nn.Module): The neural network model.
            input_shares (AdditiveSecretSharing): The shared input data.

        Returns:
            torch.Tensor: The inference result (only for the client/data owner), otherwise None.

        Examples:
            result = party.inference(net, input_shares)
        """
        for data in input_shares:
            data.party = self

        output = net(*input_shares)
        if self.party_id == 0:
            self.send(output)
        if self.party_id == 1:
            output_other = self.recv()
            output = AdditiveSecretSharing.restore_from_shares(output_other, output)
            return output.convert_to_real_field()


class PartyNeuralNetWork3PC(Party3PC):

    def dummy_model(self, *args):
        """
        Processes the input data and generates the necessary shared data.

        If client (party 0): Determines input type, sends batch quantity, and generates/sends Beaver triples.
        If server (other parties): Receives batch number and matrix triples.

        Args:
            *args: Variable length argument list. Usually contains `net` and `test_loader` for party 0.

        Returns:
            int: The batch number.
        """
        self.wait()
        num = 1

        need_triples = DEBUG_LEVEL != 2 and self.thread_model_cfg == HONEST_MAJORITY

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

            for dest_party in [1, 2]:
                self.send(dest_party, torch.tensor(num))

            if need_triples:
                mat_beaver_lists = gen_mat_beaver(
                    dummy_input=dummy_input.to('cpu'),
                    model=net,
                    num_of_triples=num,
                    num_of_party=3
                )
                local_t = mat_beaver_lists.pop()
                self.providers[RssMatmulTriples.__name__].param = local_t

                for dest_party, triples in zip([2, 1], mat_beaver_lists):
                    self.send(dest_party, triples)
        else:
            num = self.recv(0).item()

            if need_triples:
                local_t = self.recv(0)
                self.providers[RssMatmulTriples.__name__].param = local_t

        if need_triples:
            self.providers[RssMatmulTriples.__name__].load_mat_beaver()

        return num

    def inference(self, net, input_shares):
        """
        Performs inference on the neural network using shared inputs.

        Calculates communication costs during the inference process.

        Args:
            net (torch.nn.Module): The neural network model.
            input_shares: The shared input data.

        Returns:
            torch.Tensor: The restored inference result in the real field.

        Examples:
            result = party.inference(net, input_shares)
        """
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
