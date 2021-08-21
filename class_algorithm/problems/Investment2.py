def sublist_max(profits, start, end):
    # 코드를 작성하세요.

    if end - start == 0:
        if profits[start] > 0:
            return profits[start]
        else:
            return 0

    mid = (start + end) // 2
    submax_left = sublist_max(profits, start, mid)
    submax_right = sublist_max(profits, mid + 1, end)

    return max(submax_left, submax_right, submax_mid(profits, start, mid, end))


def submax_mid(profits, start, mid, end):
    submax_left = 0
    submax_right = 0
    temp = 0
    for i in range(mid, start - 1, -1):
        temp += profits[i]
        submax_left = max(temp, submax_left)

    temp = 0
    for i in range(mid + 1, end):
        temp += profits[i]
        submax_right = max(temp, submax_right)

    return submax_left + submax_right


# 테스트
list1 = [-2, -3, 4, -1, -2, 1, 5, -3]
print(sublist_max(list1, 0, len(list1) - 1))

list2 = [4, 7, -6, 9, 2, 6, -5, 7, 3, 1, -1, -7, 2]
print(sublist_max(list2, 0, len(list2) - 1))

list3 = [9, -8, 0, -7, 8, -6, -3, -8, 9, 2, 8, 3, -5, 1, -7, -1, 10, -1, -9, -5]
print(sublist_max(list3, 0, len(list3) - 1))

list4 = [-9, -8, -8, 6, -4, 6, -2, -3, -10, -8, -9, -9, 6, 2, 8, -1, -1]
print(sublist_max(list4, 0, len(list4) - 1))