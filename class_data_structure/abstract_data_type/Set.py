# 순서 상관 없이 저장
# 중복 허용 X

finished_classes = set()

# 데이터 저장
finished_classes.add("자료구조")
finished_classes.add("알고리즘")
finished_classes.add("프로그래밍 기초")
finished_classes.add("인터렉티브 웹")
finished_classes.add("데이터 사이언스")

print(finished_classes) # 세트 출력

# 중복 데이터 저장 시도
finished_classes.add("자료구조")
finished_classes.add("알고리즘")

print(finished_classes)

# 데이터 탐색
print("Is there 컴퓨터개론? {}".format("컴퓨터개론" in finished_classes))
print("자료구조" in finished_classes)

# 데이터 삭제
finished_classes.remove("자료구조")
finished_classes.remove("알고리즘")

print(finished_classes)
