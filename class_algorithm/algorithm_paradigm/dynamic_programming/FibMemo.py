def fib_memo(n, cache):
    # 코드를 작성하세요.
    if n == 1 or n == 2:
        cache[1] = 1
        cache[2] = 1
        return 1

    if cache.get(n-2) == None:
        cache[n-2] = fib_memo(n-2, cache)
    if cache.get(n-1) == None:
        cache[n-1] = fib_memo(n-1, cache)

    return cache[n-2]+cache[n-1]


def fib(n):
    # n번째 피보나치 수를 담는 사전
    fib_cache = {}

    return fib_memo(n, fib_cache)


# 테스트
print(fib(10))
print(fib(50))
print(fib(100))