import numpy as np
import random
from collections import defaultdict

class EnhancedTicTacToeAI:
    def __init__(self):
        self.board = [' '] * 9
        self.q_table = defaultdict(lambda: [0.0] * 9)
        self.alpha = 0.5
        self.alpha_decay = 0.99995
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_decay = 0.99995
        self.min_epsilon = 0.05
        self.priority_positions = {4: 0.2, 0: 0.05, 2: 0.05, 6: 0.05, 8: 0.05}
        self.win_patterns = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                             (0, 3, 6), (1, 4, 7), (2, 5, 8),
                             (0, 4, 8), (2, 4, 6)]
        self.position_threats = {
            0: [(0, 1, 2), (0, 3, 6), (0, 4, 8)],
            1: [(0, 1, 2), (1, 4, 7)],
            2: [(0, 1, 2), (2, 5, 8), (2, 4, 6)],
            3: [(3, 4, 5), (0, 3, 6)],
            4: [(3, 4, 5), (1, 4, 7), (0, 4, 8), (2, 4, 6)],
            5: [(3, 4, 5), (2, 5, 8)],
            6: [(6, 7, 8), (0, 3, 6), (2, 4, 6)],
            7: [(6, 7, 8), (1, 4, 7)],
            8: [(6, 7, 8), (2, 5, 8), (0, 4, 8)]
        }

    def reset(self):
        self.board = [' '] * 9

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == ' ']

    def check_win(self, player):
        for p in self.win_patterns:
            if self.board[p[0]] == self.board[p[1]] == self.board[p[2]] == player:
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def find_immediate_threats(self, opponent):
        threats = []
        for pos in self.available_actions():
            for pattern in self.position_threats[pos]:
                other_positions = [p for p in pattern if p != pos]
                if all(self.board[p] == opponent for p in other_positions):
                    threats.append(pos)
                    break
        return list(set(threats))

    def choose_action(self, state):
        #优先选择有威胁的地方下
        opponent = 'X'
        threats = self.find_immediate_threats(opponent)
        if threats:
            return random.choice(threats)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return self.expert_move('O')
        else:
            state_actions = self.q_table[state]
            available = self.available_actions()
            q_values = [state_actions[a] + np.random.normal(0, 0.01) for a in available]
            max_q = max(q_values)
            best_actions = [available[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def expert_move(self, player):
        opponent = 'O' if player == 'X' else 'X'
        available = self.available_actions()

        threats = []
        for pattern in self.win_patterns:
            a, b, c = pattern
            if self.board[a] == self.board[b] == opponent and self.board[c] == ' ':
                threats.append(c)
            if self.board[a] == self.board[c] == opponent and self.board[b] == ' ':
                threats.append(b)
            if self.board[b] == self.board[c] == opponent and self.board[a] == ' ':
                threats.append(a)

        if threats:
            return random.choice(list(set(threats)))

        for pos in available:
            temp = self.board.copy()
            temp[pos] = player
            if self.check_win(player):
                return pos

        priority_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        for pos in priority_order:
            if pos in available:
                return pos

        return random.choice(available)

    def get_reward(self, player):

        opponent = 'O' if player == 'X' else 'X'

        if self.check_win(player):
            return 50
        if self.check_win(opponent):
            return -50


        last_action = next((i for i, c in enumerate(self.board) if c == player), None)
        if last_action is not None:

            temp_board = self.board.copy()
            temp_board[last_action] = ' '
            was_threat = any(
                all(temp_board[p] == opponent for p in pattern if p != last_action)
                for pattern in self.position_threats[last_action]
            )
            if was_threat:
                return 20

        return 1 if self.is_draw() else 0

    def update_q_table(self, state, action, next_state, reward):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state]) if next_state in self.q_table else 0

        new_q = current_q + self.alpha * (
                reward + self.gamma * max_next_q - current_q
        )
        self.q_table[state][action] = np.clip(new_q, -50, 50)

    def train(self, episodes=10000, callback=None):
        episode_rewards = []
        win_counts = []

        for ep in range(episodes):
            self.reset()
            state = self.get_state()
            episode_reward = 0
            has_win = False

            while True:
                action = self.choose_action(state)
                self.board[action] = 'O'
                next_state = self.get_state()

                reward = self.get_reward('O')
                self.update_q_table(state, action, next_state, reward)
                episode_reward += reward

                if self.check_win('O'):
                    has_win = True
                    break
                if self.is_draw():
                    break

                opp_action = self.expert_move('X')
                self.board[opp_action] = 'X'
                next_state_opp = self.get_state()

                if self.check_win('X'):
                    self.update_q_table(state, action, next_state_opp, -15)
                    episode_reward += -15
                    break

                state = next_state_opp

            episode_rewards.append(episode_reward)
            win_counts.append(1 if has_win else 0)

            if ep % 100 == 0:
                self.alpha = max(0.1, self.alpha * 0.999)

            if callback and ep % 50 == 49:
                window = episode_rewards[-1000:]
                avg_reward = np.mean(window)

                actual_win_rate = sum(win_counts[-1000:]) / 10
                callback(ep + 1, avg_reward, actual_win_rate)

        return episode_rewards
