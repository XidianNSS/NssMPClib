import torch

from NssMPC.config import DEVICE


# a Mersenne transformation
def _torch_int32(x):
    # lsb = 0xFFFFFFFF & x
    # return lsb
    return x


def _int32(x):
    # Get the 32 least significant bits.
    return int(0xFFFFFFFF & x)


class TorchMT19937:
    def __init__(self, seeds):
        if len(seeds.shape) != 1:
            raise ValueError('seeds can only be one-dimensional arrays', )
        self.index = 624
        l = seeds.shape[0]
        s = seeds.unsqueeze(1)
        ex = torch.zeros([l, 623], dtype=torch.int32, device=DEVICE)
        self.mt = torch.cat([s, ex], dim=1)
        for i in range(1, 624):
            self.mt[:, i] = _torch_int32(1812433253 * (self.mt[:, i - 1] ^ self.mt[:, i - 1] >> 30) + i)

    def twist(self):
        for i in range(624):
            y = _torch_int32((self.mt[:, i] & 0x80000000) +
                             (self.mt[:, (i + 1) % 624] & 0x7fffffff))
            self.mt[:, i] = self.mt[:, (i + 397) % 624] ^ y >> 1

            cw = ((y % 2 != 0) + 0) * 0x9908b0df

            self.mt[:, i] = self.mt[:, i] ^ cw

        self.index = 0
        return

    def random(self, num):
        if self.index + num - 1 >= 624:
            self.twist()
        y = self.mt[:, self.index: self.index + num]
        # y = self.mt[:, self.index]

        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18
        self.index = self.index + num
        return _torch_int32(y)

    def extract_number(self):

        if self.index >= 624:
            self.twist()
        y = self.mt[:, self.index]
        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18
        self.index = self.index + 1
        return _torch_int32(y)
