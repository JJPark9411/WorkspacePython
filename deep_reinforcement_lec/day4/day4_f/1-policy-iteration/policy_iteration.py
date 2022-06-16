import numpy as np
from environment import GraphicDisplay, Env


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        # env 사이즈(5x5)만큼 테이블을 만들어 value를 0으로 초기화 (초기 가치함수)
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # env 사이즈(5x5)만큼 테이블을 만들어 각 지점에서 좌우상하로 움직일 확률을 각각 0.25로 초기화 (초기 정책)
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.policy_table[2][2] = [] # 도착지점에서의 policy는 필요없으므로 비워둠
        self.discount_factor = 0.9

    # 벨만 기대 방정식을 통해 다음 가치함수를 계산하는 정책 평가
    # v(s) = sum(PI(a|s) * (R_t+1 + GAMMA*v(s_t+1)))
    def policy_evaluation(self):
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            value = 0.0
            if state == [2, 2]: # goal 지점
                next_value_table[state[0]][state[1]] = value # value = 0
                continue

            for action in self.env.possible_actions:
                # state(x, y)에서 action(a <- action index{0, 1, 2, 3} 중 1개)을 했을 때 next_state를 리턴
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state) # value_table에서 next_state의 value를 가져옴
                # v(s) = sum(PI(a|s) * (R_t+1 + GAMMA*v(s_t+1)))
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = value
        self.value_table = next_value_table

    # 현재 가치 함수에 대해서 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]: # goal에 위치한 경우 넘어감
                continue

            value_list = []
            result = [0.0, 0.0, 0.0, 0.0] # 업데이트된 정책 변수(result) 초기화

            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            max_idx_list = np.argwhere(value_list == np.amax(value_list)) # value가 max가 되는 action이 여러 개일 수 있음
            # print('max_idx_list: ', max_idx_list)
            max_idx_list = max_idx_list.flatten()
            # print('[{},{}] max_idx_list: {}'.format(state[0], state[1], max_idx_list))
            prob = 1 / len(max_idx_list) # action이 여러 개인 경우 각 action의 확률을 계산

            for idx in max_idx_list:
                result[idx] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따라 무작위로 행동을 반환
    def get_action(self, state):
        policy = self.get_policy(state) # state에서 수행할 action의 확률을 가져옴. ex) 좌우로 움직일 수 있는 경우 [0.5, 0.5, 0, 0]
        policy = np.array(policy)
        # print('choice: ', np.random.choice(4, 1, p=policy)[0])
        return np.random.choice(4, 1, p=policy)[0]

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    # 가치 함수의 값을 반환
    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env() # environment.py에 정의되어 있음
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
