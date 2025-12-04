import os

import torch

import NssMPC.application.neural_network as nn
from NssMPC.config import NN_path
from NssMPC import PartyRuntime, SEMI_HONEST
from data.AlexNet.Alexnet import AlexNet

# set server and client address

if __name__ == '__main__':
    # 创建性能分析日志目录
    log_dir = "./profiler_logs"
    os.makedirs(log_dir, exist_ok=True)

    server = nn.party.PartyNeuralNetwork2PC(0, SEMI_HONEST)
    server.online()

    with PartyRuntime(server):

        net = AlexNet()
        net.load_state_dict(torch.load(NN_path / 'AlexNet_CIFAR10.pkl'))
        shared_param, shared_param_for_other = nn.utils.share_model(net)
        server.send(shared_param_for_other)

        num = server.dummy_model(net)
        net = nn.utils.load_model(net, shared_param)

        # 使用Profiler分析推理循环
        # with profile(
        #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(
        #             wait=1,   # 跳过第1个推理
        #             warmup=1, # 预热1个推理
        #             active=3, # 分析3个推理
        #             repeat=1
        #         ),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True,
        #         with_flops=True,         # 记录FLOPs
        #         with_modules=True        # 记录模块信息
        # ) as prof:
        #     inference_count = 0
        while num:  # and inference_count < 5:  # 限制分析次数
            shared_data = server.recv()

            # with record_function("full_inference"):
            server.inference(net, shared_data)

            # prof.step()  # 通知profiler一个步骤完成
            # inference_count += 1
            num -= 1

    server.close()

    print(f"Profiling completed. View results with: tensorboard --logdir={log_dir}")
