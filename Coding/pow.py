'''
Exponentiation by Squaring
pow(a,b)
'''


def pow(a, b):
    """
    Compute a^b using fast exponentiation (快速幂算法).

    :param a: float, base number
    :param b: int, exponent
    :return: float, result of a^b
    """
    if b == 0:
        return 1  # a^0 = 1

    if b < 0:
        a = 1 / a  # Convert to positive exponent
        b = -b

    result = 1
    while b > 0:
        if b % 2 == 1:  # If b is odd
            result *= a
        a *= a  # Square the base
        b //= 2  # Reduce exponent by half

    return result


# 测试
if __name__ == "__main__":
    print(pow(2, 10))  # 1024
    print(pow(3, 5))  # 243
    print(pow(2, -3))  # 0.125
    print(pow(5, 0))  # 1
    print(pow(1.5, 4))  # 5.0625
