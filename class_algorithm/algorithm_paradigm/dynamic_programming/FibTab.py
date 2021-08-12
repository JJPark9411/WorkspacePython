def fib_tab(n):
    # 코드를 작성하세요.

    if n < 3:
        return 1

    table = {}

    for i in range(1, n+1):
        if i < 3:
            table[i] = 1
            continue
        table[i] = table[i-1]+table[i-2]

    return table[n]


# 테스트
print(fib_tab(10))
print(fib_tab(56))
print(fib_tab(132))