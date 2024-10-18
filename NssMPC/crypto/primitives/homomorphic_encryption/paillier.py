#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import random


class Paillier(object):
    """
    Paillier Cryptosystem implementation.

    This class provides functionality for key generation, encryption, and decryption using the Paillier cryptosystem.

    ATTRIBUTES:
        * **public_key** (*tuple*): A tuple containing the public key `(n, g)`.
        * **_private_key** (*tuple*): A tuple containing the private key `(l, mu)`.

    Methods
    -------
    gen_keys(key_size=1024)
        Generates a public and private key pair of the specified bit size.

    encrypt(plaintext)
        Encrypts the given plaintext using the public key.

    decrypt(ciphertext)
        Decrypts the given ciphertext using the private key.

    encrypt_with_key(plaintext, public_key)
        Encrypts the plaintext using the provided public key.

    decrypt_with_key(ciphertext, private_key, public_key)
        Decrypts the ciphertext using the provided private key and public key.
    """

    def __init__(self):
        """
        Initializes an instance of Paillier class.

        Sets the public and private key attributes to None.
        """
        self.public_key = None
        self._private_key = None

    def gen_keys(self, key_size=1024):
        """
        Generates a key pair for Paillier.

        :param key_size: The size of the key in bits. Default is 1024 bits.
        :type key_size: int
        """
        p = _get_prime(key_size)
        q = _get_prime(key_size)
        n = p * q
        g = n + 1
        l = _lcm(p - 1, q - 1)
        mu = pow(_L(pow(g, l, n * n), n), -1, n)
        self.public_key = (n, g)
        self._private_key = (l, mu)

    def encrypt(self, plaintext):
        """
        Encrypts the given plaintext using the public key of this instance.

        :param plaintext: The plaintext to be encrypted. Can be an integer or a list of integers.
        :type plaintext: int or list
        :return: The ciphertext corresponding to the given plaintext.
        :rtype: int or list
        :raises TypeError: If the plaintext is not an integer or a list of integers.
        """
        n, g = self.public_key
        if isinstance(plaintext, int):
            return pow(g, plaintext, n * n)
        elif isinstance(plaintext, list):
            return [pow(g, plaintext[i], n * n) for i in range(len(plaintext))]
        else:
            raise TypeError('unsupported plaintext type:', type(plaintext))

    def decrypt(self, ciphertext):
        """
        Decrypts the given ciphertext using the private key of this instance.

        :param ciphertext: The ciphertext to be decrypted. Can be an integer or a list of integers.
        :type ciphertext: int or list
        :return: The plaintext corresponding to the given ciphertext.
        :rtype: int or list
        :raises TypeError: If the ciphertext is not an integer or a list of integers.
        """
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
        """
        Encrypts the plaintext using the given public key.

        :param plaintext: The plaintext to be encrypted. Can be an integer or a list of integers.
        :type plaintext: int or list
        :param public_key: The public key (n, g) to use for encryption.
        :type public_key: tuple
        :return: The ciphertext corresponding to the given plaintext.
        :rtype: int or list
        :raises TypeError: If the plaintext is not an integer or a list of integers.
        """
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
        """
        Decrypts the ciphertext using the given private and public keys.

        :param ciphertext: The ciphertext to be decrypted. Can be an integer or a list of integers.
        :type ciphertext: int or list
        :param private_key: The private key (l, mu) to use for decryption.
        :type private_key: tuple
        :param public_key: The public key (n, g) to use for decryption.
        :type public_key: tuple
        :return: The plaintext corresponding to the given ciphertext.
        :rtype: int or list
        :raises TypeError: If the ciphertext is not an integer or a list of integers.
        """
        l, mu = private_key
        n = public_key[0]
        if isinstance(ciphertext, int):
            return _L(pow(ciphertext, l, n * n), n) * mu % n
        elif isinstance(ciphertext, list):
            return [_L(pow(ciphertext[i], l, n * n), n) * mu % n for i in range(len(ciphertext))]
        else:
            raise TypeError('unsupported ciphertext type:', type(ciphertext))


def _get_prime(bits):
    """
    Generates a prime number of the specified bit length.

    :param bits: The number of bits in the prime number.
    :type bits: int
    :return: A prime number of the specified bit length.
    :rtype: int
    """
    while True:
        n = random.getrandbits(bits)
        if _is_prime(n):
            return n


def _L(x, N):
    """
    Computes the L function used in the Paillier.

    :param x: The value to be passed to the L function.
    :type x: int
    :param N: The modulus used in the L function.
    :type N: int
    :return: The result of the L function.
    :rtype: int
    """
    return (x - 1) // N


def _lcm(x, y):
    """
    Computes the least common multiple (LCM) of two integers.

    :param x: The first integer.
    :type x: int
    :param y: The second integer.
    :type y: int
    :return: The least common multiple of the two integers.
    :rtype: int
    """
    return (x * y) // _gcd(x, y)


def _gcd(x, y):
    """
    Computes the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

    :param x: The first integer.
    :type x: int
    :param y: The second integer.
    :type y: int
    :return: The greatest common divisor of the two integers.
    :rtype: int
    """
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
    """
    Determines if a number is prime using the Miller-Rabin primality test.

    :param n: The number to be tested.
    :type n: int
    :param k: The number of iterations of the test to perform. Default is 5.
    :type k: int
    :return: True if the number is prime, False otherwise.
    :rtype: bool
    """

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
