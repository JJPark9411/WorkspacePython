def trapping_rain(buildings):
    total_height = 0
    n = len(buildings)

    left = 0
    right = n - 1

    height_left = 0
    height_right = 0

    while left <= right:
        if buildings[left] < buildings[right]:
            if buildings[left] >= height_left:
                height_left = buildings[left]
            else:
                total_height += height_left - buildings[left]
            left += 1
        else:
            if buildings[right] >= height_right:
                height_right = buildings[right]
            else:
                total_height += height_right - buildings[right]
            right -= 1

    return total_height


# 테스트
print(trapping_rain([3, 0, 0, 2, 0, 4]))
print(trapping_rain([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))