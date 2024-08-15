from NssMPC import RingTensor


def zero_encoding(x: RingTensor):
    bit_len = x.bit_len
    x = x.flatten()
    zero_encoding_list = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    cw = RingTensor.where(x > 0, 0, -1)
    for i in range(bit_len - 1, -1, -1):
        current_bit = x.get_bit(i)
        cw = cw << 1
        cur_encoded = RingTensor.where(current_bit, RingTensor.random(x.shape),
                                       (x.bit_slice(i + 1, bit_len) << 1 | 1) ^ cw)
        zero_encoding_list[:, i] = cur_encoded
    fake_mask = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    for i in range(bit_len):
        fake_mask[:, i] = 1 - x.get_bit(i)
    return zero_encoding_list, fake_mask


def one_encoding(x: RingTensor):
    bit_len = x.bit_len
    x = x.flatten()
    one_encoding_list = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    cw = RingTensor.where(x > 0, 0, -1)
    for i in range(bit_len - 1, -1, -1):
        current_bit = x.get_bit(i)
        cur_encoded = RingTensor.where(current_bit, x.bit_slice(i, bit_len), RingTensor.random(x.shape))
        cw = cw << 1
        if i == 0:
            cw = 0
        one_encoding_list[:, i] = cur_encoded ^ cw
    fake_mask = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    for i in range(bit_len):
        fake_mask[:, i] = x.get_bit(i)
    return one_encoding_list, fake_mask
