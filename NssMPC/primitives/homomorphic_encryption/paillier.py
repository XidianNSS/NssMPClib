#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import random


class Paillier(object):
    """Paillier Cryptosystem implementation.

    This class provides functionality for key generation, encryption, and decryption using the Paillier cryptosystem.

    Attributes:
        public_key (tuple): A tuple containing the public key (n, g).
        _private_key (tuple): A tuple containing the private key (l, mu).

    Examples:
        >>> paillier = Paillier()
        >>> paillier.gen_keys()
        >>> enc = paillier.encrypt(123)
        >>> dec = paillier.decrypt(enc)
    """

    def __init__(self):
        """Initializes an instance of Paillier class.

        Sets the public and private key attributes to None.
        """
        self.public_key = None
        self._private_key = None

    def gen_keys(self, key_size=1024):
        """Generates a key pair for Paillier.

        Args:
            key_size (int, optional): The size of the key in bits. Defaults to 1024.

        Examples:
            >>> paillier.gen_keys(1024)
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
        """Encrypts the given plaintext using the public key.

        Args:
            plaintext (int or list): The plaintext to encrypt. Can be a single integer or a list of integers.

        Returns:
            int or list: The encrypted ciphertext. Returns a single integer if input is int, or a list if input is list.

        Raises:
            TypeError: If the plaintext type is not supported.

        Examples:
            >>> ciphertext = paillier.encrypt(10)
        """
        n, g = self.public_key
        if isinstance(plaintext, int):
            return pow(g, plaintext, n * n)
        elif isinstance(plaintext, list):
            return [pow(g, plaintext[i], n * n) for i in range(len(plaintext))]
        else:
            raise TypeError('unsupported plaintext type:', type(plaintext))

    def decrypt(self, ciphertext):
        """Decrypts the given ciphertext using the private key.

        Args:
            ciphertext (int or list): The ciphertext to decrypt. Can be a single integer or a list of integers.

        Returns:
            int or list: The decrypted plaintext. Returns a single integer if input is int, or a list if input is list.

        Raises:
            TypeError: If the ciphertext type is not supported.

        Examples:
            >>> plaintext = paillier.decrypt(ciphertext)
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
        """Encrypts the plaintext using the provided public key.

        Args:
            plaintext (int or list): The plaintext to encrypt.
            public_key (tuple): The public key to use for encryption (n, g).

        Returns:
            int or list: The encrypted ciphertext.

        Raises:
            TypeError: If the plaintext type is not supported.

        Examples:
            >>> enc = Paillier.encrypt_with_key(10, public_key)
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
        """Decrypts the ciphertext using the provided private key and public key.

        Args:
            ciphertext (int or list): The ciphertext to decrypt.
            private_key (tuple): The private key to use for decryption (l, mu).
            public_key (tuple): The public key to use for decryption (n, g).

        Returns:
            int or list: The decrypted plaintext.

        Raises:
            TypeError: If the ciphertext type is not supported.

        Examples:
            >>> dec = Paillier.decrypt_with_key(enc, private_key, public_key)
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
    """Generates a prime number of the specified bit length.

    Args:
        bits (int): The number of bits in the prime number.

    Returns:
        int: A prime number of the specified bit length.
    """
    while True:
        n = random.getrandbits(bits)
        if _is_prime(n):
            return n


def _L(x, N):
    """Computes the L function used in the Paillier cryptosystem.

    Args:
        x (int): The value to be passed to the L function.
        N (int): The modulus used in the L function.

    Returns:
        int: The result of the L function.
    """
    return (x - 1) // N


def _lcm(x, y):
    """Computes the least common multiple (LCM) of two integers.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        int: The least common multiple of the two integers.
    """
    return (x * y) // _gcd(x, y)


def _gcd(x, y):
    """Computes the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        int: The greatest common divisor of the two integers.
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
    """Determines if a number is prime using the Miller-Rabin primality test.

    Args:
        n (int): The number to be tested.
        k (int, optional): The number of iterations of the test to perform. Defaults to 5.

    Returns:
        bool: True if the number is prime, False otherwise.
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
