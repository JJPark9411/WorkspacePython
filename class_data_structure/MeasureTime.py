# 예시를 위해 사용할 모듈 import
import time

# 데이터를 리스트에 저장한다
test_list = [x for x in range(0, 1000000)]

# 특정 항목이 리스트에 있는지 확인할 때 걸리는 시간 파악
t_0 = time.time()
999999 in test_list # 리스트 탐색
t_1 = time.time()

print(f"리스트에서 특정 항목을 찾는데 걸린 시간: {t_1 - t_0}")

# 데이터를 set에 저장한다
test_set = set([x for x in range(0, 1000000)])

# 특정 항목이 set에 있는지 확인할 때 걸리는 시간 파악
t_0 = time.time()
999999 in test_set
t_1 = time.time()

print(f"세트에서 특정 항목을 찾는데 걸린 시간: {t_1 - t_0}")