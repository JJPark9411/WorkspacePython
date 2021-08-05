grades = {}

# key - value 데이터 삽입
grades["현승"] = 80
grades["태호"] = 60
grades["영훈"] = 90
grades["지웅"] = 70
grades["동욱"] = 96

print(grades) # 딕셔너리 출력

# 하나의 key에 여러 value 저장 시도
grades["태호"] = 100

print(grades)

# key를 이용해서 value 탐색
print("동욱:", grades["동욱"])
print("지웅: {}".format(grades["지웅"]))

# key를 이용한 삭제
grades.pop("동욱")
grades.pop("지웅")

print(grades)