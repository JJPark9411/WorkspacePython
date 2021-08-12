def max_profit(price_list, count):
    # 코드를 작성하세요.
    max_profit_n = [0, 100]

    temp = 0
    for i in range(2, count+1):
        for j in range(1, (i//2)+1):
            temp = max(max_profit_n[j]+max_profit_n[i-j], temp)
            # print(max_profit_n)
            # print("[%d, %d] temp: %d"%(j, i-j, temp))
        max_profit_n.append(max(temp, price_list[i-1]))

    return max_profit_n[count]


# 테스트
print(max_profit([100, 400, 800, 900, 1000], 5))