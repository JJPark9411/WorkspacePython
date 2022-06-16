import numpy as np
import random
from collections import defaultdict
from environment import Env


# 몬테카를로 에이전트 (모든 에피소드 각각의 샘플로 부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = [] # episode에서 뽑은 샘플을 저장
        self.value_table = defaultdict(float)

    # 메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트
    # episode가 종료된 시점에 수행
    def update(self):
        G_t = 0
        visit_state = []
        for reward in reversed(self.samples): # 종료된 시점부터 계산하기 위해 reversed를 사용
            state = str(reward[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = reward[1] + self.discount_factor * G_t
                value = self.value_table[state]
                self.value_table[state] = (value + self.learning_rate * (G_t - value))

    # 큐 함수에 따라서 행동을 반환
    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 랜덤 행동
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # 가능한 다음 모든 상태들을 반환
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state


# 메인 함수
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions))) # 상하좌우 action에 대한 list를 만듦. (0, 1, 2, 3) n_actions=4

    for episode in range(1000):
        state = env.reset() # 초기 state(시작점) 반환
        action = agent.get_action(state)

        while True:
            env.render()

            next_state, reward, done = env.step(action)
            agent.save_sample(next_state, reward, done)

            action = agent.get_action(next_state)

            # epsiode가 종료되면 큐함수 업데이트
            if done:
                print('episode: ', episode)
                agent.update()
                agent.samples.clear()
                break
