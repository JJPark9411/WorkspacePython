def binary_search(element, some_list):
    # 코드를 작성하세요.
    front = 0
    rear = len(some_list)
    mid = (front+rear)//2

    while(True):
        if some_list[mid] == element:
            return mid
        elif some_list[mid] < element:
            front = mid+1
            mid = (front+rear)//2
        elif some_list[mid] > element:
            rear = mid-1
            mid = (front+rear)//2

        if front > rear:
            return None

print(binary_search(2, [2, 3, 5, 7, 11]))
print(binary_search(0, [2, 3, 5, 7, 11]))
print(binary_search(5, [2, 3, 5, 7, 11]))
print(binary_search(3, [2, 3, 5, 7, 11]))
print(binary_search(11, [2, 3, 5, 7, 11]))