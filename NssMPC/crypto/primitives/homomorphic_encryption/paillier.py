import random


class Paillier(object):
    def __init__(self):
        self.public_key = None
        self._private_key = None

    def gen_keys(self, key_size=1024):
        p = _get_prime(key_size)
        q = _get_prime(key_size)
        n = p * q
        g = n + 1
        l = _lcm(p - 1, q - 1)
        mu = pow(_L(pow(g, l, n * n), n), -1, n)
        self.public_key = (n, g)
        self._private_key = (l, mu)

    def encrypt(self, plaintext):
        n, g = self.public_key
        if isinstance(plaintext, int):
            return pow(g, plaintext, n * n)
        elif isinstance(plaintext, list):
            return [pow(g, plaintext[i], n * n) for i in range(len(plaintext))]
        else:
            raise TypeError('unsupported plaintext type:', type(plaintext))

    def decrypt(self, ciphertext):
        l, mu = self._private_key
        n = self.public_key[0]
        if isinstance(ciphertext, int):
            return _L(pow(ciphertext, l, n * n), n) * mu % n
        elif isinstance(ciphertext, list):
            return [_L(pow(ciphertext[i], l, n * n), n) * mu % n for i in range(len(ciphertext))]
        else:
            raise TypeError('unsupported ciphertext type:', type(ciphertext))

    @staticmethod
    def encrypt_with_key(plaintext, public_key):
        # TODO: generate a random number r
        n, g = public_key
        if isinstance(plaintext, int):
            return pow(g, plaintext, n * n)
        elif isinstance(plaintext, list):
            return [pow(g, plaintext[i], n * n) for i in range(len(plaintext))]
        else:
            raise TypeError('unsupported plaintext type:', type(plaintext))

    @staticmethod
    def decrypt_with_key(ciphertext, private_key, public_key):
        l, mu = private_key
        n = public_key[0]
        if isinstance(ciphertext, int):
            return _L(pow(ciphertext, l, n * n), n) * mu % n
        elif isinstance(ciphertext, list):
            return [_L(pow(ciphertext[i], l, n * n), n) * mu % n for i in range(len(ciphertext))]
        else:
            raise TypeError('unsupported ciphertext type:', type(ciphertext))


def _get_prime(bits):
    while True:
        n = random.getrandbits(bits)
        if _is_prime(n):
            return n


def _L(x, N):
    return (x - 1) // N


def _lcm(x, y):
    return (x * y) // _gcd(x, y)


def _gcd(x, y):
    if x < y:
        temp = x
        x = y
        y = temp
    while x % y != 0:
        r = x % y
        x = y
        y = r
    return y


def _is_prime(n, k=5):
    def witness(a, n):
        t = 0
        u = n - 1
        while u % 2 == 0:
            t += 1
            u //= 2
            x = pow(a, u, n)
            if x == 1 or x == n - 1:
                return True
            for _ in range(t - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    return True
                return False

    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        a = random.randint(2, n - 2)
        if not witness(a, n):
            return False
        return True
