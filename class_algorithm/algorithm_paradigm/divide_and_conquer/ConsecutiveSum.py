def consecutive_sum(start, end):
    # 코드를 작성하세요
    if start == end:
        return start
    mid = (start+end)//2
    return consecutive_sum(start, mid)+consecutive_sum(mid+1, end)


# 테스트
print(consecutive_sum(1, 10))
print(consecutive_sum(1, 100))
print(consecutive_sum(1, 253))
print(consecutive_sum(1, 388))