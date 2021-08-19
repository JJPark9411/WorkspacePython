def power(x, y):
    # 코드를 작성하세요.
    if y == 0:
        return 1
    elif y == 1:
        return x

    p = y // 2
    r = y % 2

    num = power(x, p)
    if r == 0:
        return num * num
    elif r == 1:
        return num * num * x


# 테스트
print(power(3, 5))
print(power(5, 6))
print(power(7, 9))