# 높이 n개의 계단을 올라가는 방법을 리턴한다
def staircase(stairs, possible_steps):
    memo = {}
    memo[0] = 1
    memo[1] = 1

    return staircase_memo(stairs, possible_steps, memo)


def staircase_memo(stairs, possible_steps, memo):
    if stairs < 2:
        return memo[stairs]

    if stairs in memo:
        return memo[stairs]

    sum = 0
    for step in possible_steps:
        prev_stair = stairs - step
        if prev_stair < 0:
            continue
        else:
            sum += staircase_memo(prev_stair, possible_steps, memo)

    return sum


print(staircase(5, [1, 2, 3]))
print(staircase(6, [1, 2, 3]))
print(staircase(7, [1, 2, 4]))
print(staircase(8, [1, 3, 5]))