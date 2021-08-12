def max_profit_memo(price_list, count, cache):
    # 코드를 작성하세요.
    if count == 1:
        return price_list[1]

    if count in cache:
        return cache[count]

    temp = 0
    for i in range(2, count+1):
        for j in range(1, (i//2)+1):
            # print("count: %d" % count)
            if len(price_list) > i:
                temp = max(max_profit_memo(price_list, j, cache)+max_profit_memo(price_list, i-j, cache), temp, price_list[i])
            else:
                temp = max(max_profit_memo(price_list, j, cache) + max_profit_memo(price_list, i - j, cache), temp)
            # print("[%d+%d=%d] temp: %d"%(j, i-j, i, temp))

    cache[count] = temp
    # print(cache)

    return cache[count]


def max_profit(price_list, count):
    max_profit_cache = {}

    return max_profit_memo(price_list, count, max_profit_cache)


# 테스트
print(max_profit([0, 100, 400, 800, 900, 1000], 5))
print(max_profit([0, 100, 400, 800, 900, 1000], 10))
print(max_profit([0, 100, 400, 800, 900, 1000, 1400, 1600, 2100, 2200], 9))
