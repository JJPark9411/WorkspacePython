def max_profit(stock_list):
    # 코드를 작성하세요.
    buy = stock_list[0]
    sell = stock_list[1]
    pivot = 999
    for i in range(2, len(stock_list)):
        if stock_list[i] - pivot >= sell - buy:
            buy = pivot
            sell = stock_list[i]
            pivot = 999

        if stock_list[i] > sell:
            if buy > sell:
                buy = sell
            sell = stock_list[i]
        else:
            if stock_list[i] < buy:
                pivot = stock_list[i]

        # print("buy: %d\nsell: %d\n"%(buy, sell))

    return sell - buy


# 테스트
print(max_profit([7, 1, 5, 3, 6, 4]))
print(max_profit([7, 6, 4, 3, 1]))
print(max_profit([11, 13, 9, 13, 20, 14, 19, 12, 19, 13]))
print(max_profit([12, 4, 11, 18, 17, 19, 1, 19, 14, 13, 7, 15, 10, 1, 3, 6]))