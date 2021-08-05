# 파라미터 some_list를 거꾸로 뒤집는 함수
def flip(some_list):
    # 코드를 입력하세요.
    if len(some_list) == 0 or len(some_list) == 1:
        return some_list
    front = some_list[0]
    rear = some_list[len(some_list)-1]
    temp = some_list[1:len(some_list)-1]
    return [rear] + flip(temp) + [front]


# 테스트
some_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
some_list = flip(some_list)
print(some_list)