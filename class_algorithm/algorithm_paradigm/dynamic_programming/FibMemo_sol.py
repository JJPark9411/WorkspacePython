def fib_memo(n, cache):
    # base case
    if n < 3:
        return 1

    # 이미 n번째 피보나치를 계산했으면:
    # 저장된 값을 바로 리턴한다
    if n in cache:
        return cache[n]

    # 아직 n번째 피보나치 수를 계산하지 않았으면:
    # 계산을 한 후 cache에 저장
    cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)

    # 계산한 값을 리턴한다
    return cache[n]


def fib(n):
    # n번째 피보나치 수를 담는 사전
    fib_cache = {}

    return fib_memo(n, fib_cache)


# 테스트
print(fib(10))
print(fib(50))
print(fib(100))