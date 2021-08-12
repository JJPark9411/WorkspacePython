def max_profit(price_list, count):
    # 코드를 작성하세요.
    if count < 2:
        return price_list[count]

    if count < len(price_list):
        profit = price_list[count]
    else:
        profit = 0

    for i in range(2, count+1):
        for j in range(1, (i//2)+1):
            profit = max(max_profit(price_list, j)+max_profit(price_list, i-j), profit)

    return profit


# 테스트
print(max_profit([0, 200, 600, 900, 1200, 2000], 5))
print(max_profit([0, 300, 600, 700, 1100, 1400], 8))
print(max_profit([0, 100, 200, 400, 600, 900, 1200, 1300, 1500, 1800], 9))
