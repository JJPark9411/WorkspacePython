def min_fee(pages_to_print):
    # 코드를 작성하세요.
    pages_to_print.sort()

    fee = 0
    fee_total = 0
    for page in pages_to_print:
        fee += page
        fee_total += fee

    return fee_total


# 테스트
print(min_fee([6, 11, 4, 1]))
print(min_fee([3, 2, 1]))
print(min_fee([3, 1, 4, 3, 2]))
print(min_fee([8, 4, 2, 3, 9, 23, 6, 8]))
