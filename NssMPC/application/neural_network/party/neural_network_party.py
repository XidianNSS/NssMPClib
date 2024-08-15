import torch
import torch.utils.data

from NssMPC.application.neural_network.utils.converter import gen_mat_beaver, image2tensor
from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from NssMPC.crypto.aux_parameter import MatmulTriples
from NssMPC.secure_model.mpc_party import SemiHonest3PCParty, HonestMajorityParty, SemiHonestCS
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *


class NeuralNetworkCS(SemiHonestCS):
    """
    The neural network party based on client-server architecture

    Server as model owner and provider
    Client as data owner and provider
    """

    def __init__(self, type):
        super(NeuralNetworkCS, self).__init__(type)
        self.set_multiplication_provider()
        self.set_comparison_provider()
        self.set_nonlinear_operation_provider()

    def activate_by_gelu(self):
        """
        Use the GeLU activation function.
        Returns:
        """
        from NssMPC.crypto.aux_parameter import GeLUKey
        self.append_provider(ParamProvider(param_type=GeLUKey))

    def dummy_model(self, inputs):
        """
        Simulate the model.
        Generate in advance the parameters needed like the triples for the matrix multiplication.
        TODO: It will need to be modified to a trusted third party in subsequent steps

        Args:
            inputs: the input to the model provider is the class of the model,
                    while the input to the data provider is the data to be inferred.
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
                num = len(inputs.dataset) // inputs.batch_size  # TODO DATASET may not have a len method
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
        Infer in ciphertext

        Args:
            net: the neural network model
            input_shares: The shared input data.

        Returns:
            The data owner(client) output the inference result.
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
    def __init__(self, id):
        super().__init__(id)


class HonestMajorityNeuralNetWork3PC(HonestMajorityParty):
    def __init__(self, party_id):
        super().__init__(party_id)

    def dummy_model(self, *args):
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
                num = len(test_loader.dataset) // test_loader.batch_size  # TODO DATASET may not have a len method
                dummy_input = torch.ones_like(enumerate(test_loader).__next__()[1][0])
            else:
                raise TypeError("unsupported data type:", type(test_loader))

            self.send(1, torch.tensor(num))
            self.send(2, torch.tensor(num))

            if DEBUG_LEVEL != 2:
                mat_beaver_lists = gen_mat_beaver(dummy_input=dummy_input.to('cpu'), model=net, num_of_triples=num,
                                                  num_of_party=3)
                local_t = mat_beaver_lists.pop()
                # a, b, c = local_t[0]
                # a.party = self
                # b.party = self
                # c.party = self

                self.providers[RssMatmulTriples.__name__].param = local_t
                self.send(2, mat_beaver_lists.pop())
                self.send(1, mat_beaver_lists.pop())



        else:
            num = self.receive(0).item()
            if DEBUG_LEVEL != 2:
                local_t = self.receive(0)
                # a, b, c = local_t[0]
                # a.party = self
                # b.party = self
                # c.party = self
                self.providers[RssMatmulTriples.__name__].param = local_t
        # print("a", a)
        # print("b", b)
        # print("c", c)

        # print("res ", a.restore() @ b.restore())

        self.providers[RssMatmulTriples.__name__].load_mat_beaver()
        return num

    def inference(self, net, input_shares):
        pass

    def wait(self):
        """
        Ensure synchronisation of communications between the two parties
        """
        self.send((self.party_id + 1) % 3, 0)
        self.send((self.party_id + 2) % 3, 0)
        self.receive((self.party_id + 1) % 3)
        self.receive((self.party_id + 2) % 3)
