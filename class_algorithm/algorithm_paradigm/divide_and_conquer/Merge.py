def merge(list1, list2):
    # 코드를 작성하세요.
    combined_list = []
    while(True):
        if list1 == [] and list2 == []:
            return combined_list
        elif list1 == []:
            return combined_list+list2
        elif list2 == []:
            return combined_list+list1
        else:
            if list1[0] <= list2[0]:
                combined_list.append(list1[0])
                list1.remove(list1[0])
            else:
                combined_list.append(list2[0])
                list2.remove(list2[0])


# 테스트
print(merge([1], []))
print(merge([], [1]))
print(merge([2], [1]))
print(merge([1, 2, 3, 4], [5, 6, 7, 8]))
print(merge([5, 6, 7, 8], [1, 2, 3, 4]))
print(merge([4, 7, 8, 9], [1, 3, 6, 10]))