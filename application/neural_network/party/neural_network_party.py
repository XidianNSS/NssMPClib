import torch
import torch.utils.data

from application.neural_network.model.model_converter import gen_mat_beaver, image2tensor
from config.base_configs import DEBUG_LEVEL
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.matrix_triples import MatrixTriples
from model.mpc.semi_honest_party import SemiHonestCS


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
                self.providers[MatrixTriples].param = mat_beaver_lists.pop()
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
                self.providers[MatrixTriples].param = self.receive()

        self.providers[MatrixTriples].load_mat_beaver()
        return num

    def inference(self, net, input_shares: ArithmeticSharedRingTensor):
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
            output = ArithmeticSharedRingTensor.restore_from_shares(output_other, output)
            return output.convert_to_real_field()
