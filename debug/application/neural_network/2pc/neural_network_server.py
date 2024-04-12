from application.neural_network.model.model_converter import share_model, load_secure_model
from application.neural_network.party.neural_network_party import NeuralNetworkCS
from config.base_configs import *
from config.network_configs import *
from data.neural_network.AlexNet.Alexnet import AlexNet

# set server and client address
server_server_address = (SERVER_IP, SERVER_SERVER_PORT)
server_client_address = (SERVER_IP, SERVER_CLIENT_PORT)

client_server_address = (CLIENT_IP, CLIENT_SERVER_PORT)
client_client_address = (CLIENT_IP, CLIENT_CLIENT_PORT)

if __name__ == '__main__':
    server = NeuralNetworkCS(type='server')
    server.connect(server_server_address, server_client_address, client_server_address, client_client_address)

    net = AlexNet()

    net.load_state_dict(torch.load("data/neural_network/AlexNet/MNIST_bak.pkl"))

    shared_param, shared_param_for_other = share_model(net)
    server.send(shared_param_for_other)

    num = server.dummy_model(net)

    net = load_secure_model(net, shared_param)

    while num:
        shared_data = server.receive()
        server.inference(net, shared_data)
        num -= 1
    server.close()
