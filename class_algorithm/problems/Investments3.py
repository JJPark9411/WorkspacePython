def sublist_max(profits):
    # 코드를 작성하세요.
    sum_max = 0
    sum = 0
    for num in profits:
        sum += num
        if sum < 0:
            sum = 0
            continue

        sum_max = sum_max if sum_max > sum else sum

    return sum_max


# 테스트
print(sublist_max([7, -3, 4, -8]))
print(sublist_max([-2, -3, 4, -1, -2, 1, 5, -3, -1]))