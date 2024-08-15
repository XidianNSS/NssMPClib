from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.common.ring.ring_tensor import RingTensor
import time

num = 1280000000
a = RingTensor.random([num])
total_bit = num * a.bit_len
B = total_bit / 8
KB = B / 1024
MB = KB / 1024
print("total bits:", total_bit)
print("total bytes:", B)
print("total KB:", KB)
print("total MB:", MB)

S = SemiHonestCS('server')
S.online()

print("num of elements:", num)
# convert to MB

total_time = 0
for i in range(6):
    print("start sending")
    s_t = time.time()
    S.send(a)
    e_t = time.time()
    print("sending time:", e_t - s_t)
    print("end sending ")
    if i == 0:
        continue
    total_time += e_t - s_t

print("total time:", total_time)
print("avg time:", total_time / 5)

# total_time = 0
# for i in range(6):
#     print("start receiving")
#     s_t = time.time()
#     b = S.receive()
#     e_t = time.time()
#     print("receiving time:", e_t - s_t)
#     print("end sending ")
#     if i == 0:
#         continue
#     total_time += e_t - s_t
#
# print("total time:", total_time)
# print("avg time:", total_time / 5)
