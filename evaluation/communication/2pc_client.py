from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.common.ring.ring_tensor import RingTensor
import time

num = 500000000

b = RingTensor.random([num])
print("num of elements:", num)
# convert to MB
total_bit = num * b.bit_len
B = total_bit / 8
KB = B / 1024
MB = KB / 1024
print("total bits:", total_bit)
print("total bytes:", B)
print("total KB:", KB)
print("total MB:", MB)

C = SemiHonestCS('client')
C.online()
# ttt = 0
# for i in range(5):
#     print("start receiving")
#     s_t = time.time()
#     a = C.receive()
#     e_t = time.time()
#     print("receiving time:", e_t - s_t)
#     ttt += e_t - s_t
#     print("end receiving")
# print(ttt)
# print(ttt/5)

print("start sending")
ttt = 0
for i in range(5):
    print("start sending")
    s_t = time.time()
    C.send(b)
    e_t = time.time()
    print("sending time:", e_t - s_t)
    ttt += e_t - s_t
    print("end sending")
print(ttt)
print(ttt / 5)

# C.close()
