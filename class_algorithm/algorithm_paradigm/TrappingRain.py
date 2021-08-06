def trapping_rain(buildings):
    # 코드를 작성하세요
    sum = 0
    for i in range(1, len(buildings)-1):
        max_left = buildings[i]
        max_right = buildings[i]
        for left in range(i):
            max_left = max(max_left, buildings[left])
        for right in range(i+1, len(buildings)):
            max_right = max(max_right, buildings[right])
        height = min(max_left, max_right)
        rain = height - buildings[i]
        if rain > 0:
            # print("[%d] %d" % (i, rain))
            sum += rain
    return sum

# 테스트
print(trapping_rain([3, 0, 0, 2, 0, 4]))
print(trapping_rain([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))