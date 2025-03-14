    import os
    import random
    import copy
    import csv
    import numpy as np
    from collections import deque

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    # =============================================================================
    # Simplified Full Poker Environment with Opponent Policy (with Belief State)
    # =============================================================================
    class FullPokerEnvWithOpponentPolicy:
        def __init__(self, config=None):
            self.num_players = 6
            self.agent_id = 0
            self.action_list = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
            # Using two suits (resulting in 26 cards)
            self.suits = ['H', 'S']
            self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            self.full_deck = [r + s for s in self.suits for r in self.ranks]

            self.small_blind_amount = 50
            self.big_blind_amount = 100

            self.dealer = -1  # so that on first reset, dealer becomes 0

            self.pot = 0
            self.stacks = {pid: 10000 for pid in range(self.num_players)}
            self.hands = {}
            self.community_cards = []
            self.current_bets = {}
            self.current_max_bet = 0
            self.active_players = []
            self.players_to_act = []
            self.stage = None  # stages: preflop, flop, turn, river, showdown
            self.round_over = False

            self.opponent_policies = {pid: None for pid in range(1, self.num_players)}

            self.max_steps = 100
            self.steps_taken = 0

        def reset(self):
            self.steps_taken = 0
            self.stacks = {pid: 10000 for pid in range(self.num_players)}
            self.deck = self.full_deck.copy()
            random.shuffle(self.deck)

            self.community_cards = []
            self.pot = 0
            self.hands = {}
            for pid in range(self.num_players):
                self.hands[pid] = [self.deck.pop(), self.deck.pop()]

            self.dealer = (self.dealer + 1) % self.num_players
            self.small_blind = (self.dealer + 1) % self.num_players
            self.big_blind = (self.dealer + 2) % self.num_players

            self.current_bets = {pid: 0 for pid in range(self.num_players)}
            self.active_players = [pid for pid in range(self.num_players) if self.stacks[pid] > 0]

            sb = self.small_blind
            bb = self.big_blind
            sb_bet = min(self.small_blind_amount, self.stacks[sb])
            self.stacks[sb] -= sb_bet
            self.current_bets[sb] = sb_bet
            self.pot += sb_bet

            bb_bet = min(self.big_blind_amount, self.stacks[bb])
            self.stacks[bb] -= bb_bet
            self.current_bets[bb] = bb_bet
            self.pot += bb_bet
            self.current_max_bet = bb_bet

            self.stage = "preflop"
            self.round_over = False

            start = (bb + 1) % self.num_players
            self.players_to_act = []
            p = start
            while True:
                if p in self.active_players:
                    self.players_to_act.append(p)
                if p == bb:
                    break
                p = (p + 1) % self.num_players
            self.current_player = self.players_to_act[0] if self.players_to_act else None
            return self._get_obs(self.agent_id)

        def _compute_belief(self, player_id):
            # Compute a belief state for the agent about each opponent's hole cards.
            known = set(self.hands[player_id]) | set(self.community_cards)
            belief = {}
            for opp in range(self.num_players):
                if opp != player_id:
                    # For simplicity, the belief is all cards in full_deck not in known.
                    belief[opp] = [card for card in self.full_deck if card not in known]
            return belief

        def _get_obs(self, player_id):
            obs = {
                'hand': self.hands[player_id],
                'community_cards': self.community_cards,
                'pot': self.pot,
                'current_bet': self.current_max_bet,
                'player_stack': self.stacks[player_id],
                'player_current_bet': self.current_bets[player_id],
                'stage': self.stage,
                'legal_actions': self._get_legal_actions(player_id),
                'beliefs': self._compute_belief(player_id)
            }
            return obs

        def _get_legal_actions(self, player_id):
            if player_id not in self.active_players:
                return []
            legal = []
            player_bet = self.current_bets[player_id]
            if player_bet < self.current_max_bet:
                legal.extend(['fold', 'call', 'bet_small', 'bet_big', 'all_in'])
            else:
                legal.extend(['check', 'bet_small', 'bet_big', 'all_in', 'fold'])
            return legal

        def step(self, action):
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.stage = "showdown"

            if self.current_player == self.agent_id:
                self._process_action(self.agent_id, action)

            self._update_players_to_act(self.current_player)

            while self.current_player is not None and self.current_player != self.agent_id:
                self.steps_taken += 1
                if self.steps_taken >= self.max_steps:
                    self.stage = "showdown"
                    break
                opp_action = self._select_opponent_action(self.current_player)
                self._process_action(self.current_player, opp_action)
                self._update_players_to_act(self.current_player)
                if not self.players_to_act:
                    self._progress_stage()
                    if self.steps_taken >= self.max_steps or self.stage == "showdown":
                        self.stage = "showdown"
                        break

            if self.current_player is None and self.stage != "showdown":
                self._progress_stage()

            if self.stage == "showdown":
                obs, reward, done, info = self._finalize_hand()
                return obs, reward, done, info

            obs = self._get_obs(self.agent_id)
            return obs, 0, False, {}

        def _process_action(self, player, action):
            if self.stacks[player] <= 0:
                if player in self.active_players:
                    self.active_players.remove(player)
                return

            if action == 'fold':
                if player in self.active_players:
                    self.active_players.remove(player)
                self.current_bets[player] = 0
            elif action in ['call', 'check']:
                call_amount = self.current_max_bet - self.current_bets[player]
                amount = min(call_amount, self.stacks[player])
                self.stacks[player] -= amount
                self.current_bets[player] += amount
                self.pot += amount
            elif action == 'bet_small':
                raise_amount = 50
                new_bet = self.current_max_bet + raise_amount
                amount_to_call = new_bet - self.current_bets[player]
                amount = min(amount_to_call, self.stacks[player])
                self.stacks[player] -= amount
                self.current_bets[player] += amount
                self.pot += amount
                if new_bet > self.current_max_bet:
                    self.current_max_bet = self.current_bets[player]
                    self.players_to_act = [p for p in self.active_players if p != player]
            elif action == 'bet_big':
                raise_amount = 100
                new_bet = self.current_max_bet + raise_amount
                amount_to_call = new_bet - self.current_bets[player]
                amount = min(amount_to_call, self.stacks[player])
                self.stacks[player] -= amount
                self.current_bets[player] += amount
                self.pot += amount
                if new_bet > self.current_max_bet:
                    self.current_max_bet = self.current_bets[player]
                    self.players_to_act = [p for p in self.active_players if p != player]
            elif action == 'all_in':
                amount = self.stacks[player]
                self.stacks[player] = 0
                self.current_bets[player] += amount
                self.pot += amount
                if self.current_bets[player] > self.current_max_bet:
                    self.current_max_bet = self.current_bets[player]
                    self.players_to_act = [p for p in self.active_players if p != player]
            if player in self.players_to_act:
                self.players_to_act.remove(player)

        def _update_players_to_act(self, current):
            if self.players_to_act:
                self.current_player = self.players_to_act[0]
            else:
                self.current_player = None

        def _progress_stage(self):
            for pid in self.active_players:
                self.current_bets[pid] = 0
            self.current_max_bet = 0
            self.players_to_act = []

            if self.stage == "preflop":
                self.community_cards.extend([self.deck.pop() for _ in range(3)])
                self.stage = "flop"
            elif self.stage == "flop":
                self.community_cards.append(self.deck.pop())
                self.stage = "turn"
            elif self.stage == "turn":
                self.community_cards.append(self.deck.pop())
                self.stage = "river"
            elif self.stage == "river":
                self.stage = "showdown"

            start = (self.dealer + 1) % self.num_players
            order = []
            p = start
            while len(order) < len(self.active_players):
                if p in self.active_players:
                    order.append(p)
                p = (p + 1) % self.num_players
            self.players_to_act = order
            self.current_player = self.players_to_act[0] if self.players_to_act else None

        def _finalize_hand(self):
            scores = {}
            for pid in self.active_players:
                full_hand = self.hands[pid] + self.community_cards
                scores[pid] = evaluate_hand(full_hand)

            winners = []
            best_score = None
            for pid, score in scores.items():
                if best_score is None or score > best_score:
                    best_score = score
                    winners = [pid]
                elif score == best_score:
                    winners.append(pid)
            win_amount = self.pot / len(winners) if winners else 0
            for pid in winners:
                self.stacks[pid] += win_amount
            agent_reward = win_amount if self.agent_id in winners else -self.current_bets[self.agent_id]
            self.round_over = True
            obs = self._get_obs(self.agent_id)
            info = {'winners': winners, 'scores': scores}
            return obs, agent_reward, True, info

        def _select_opponent_action(self, player):
            if self.opponent_policies[player] is not None:
                obs = self._get_obs(player)
                return self.opponent_policies[player](obs)
            else:
                legal = self._get_legal_actions(player)
                return random.choice(legal) if legal else 'fold'

    # =============================================================================
    # A Very Simplified Poker Hand Evaluator
    # =============================================================================
    def evaluate_hand(cards):
        rank_map = {r: i for i, r in enumerate(
            ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'], start=2)}
        parsed = []
        for card in cards:
            if len(card) == 3:
                rank = card[:2]
                suit = card[2]
            else:
                rank = card[0]
                suit = card[1]
            parsed.append((rank_map[rank], suit))
        ranks = [r for r, s in parsed]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)
        suit_counts = {}
        for r, s in parsed:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        flush = any(count >= 5 for count in suit_counts.values())
        unique_ranks = sorted(set(ranks))
        straight = False
        straight_high = 0
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                straight = True
                straight_high = unique_ranks[i+4]
        if flush and straight:
            hand_rank = 9
            tiebreaker = straight_high
        elif 4 in counts:
            hand_rank = 8
            quad = max(r for r, cnt in rank_counts.items() if cnt == 4)
            kicker = max(r for r in ranks if r != quad)
            tiebreaker = (quad, kicker)
        elif 3 in counts and 2 in counts:
            hand_rank = 7
            triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
            pair = max(r for r, cnt in rank_counts.items() if cnt >= 2 and r != triple)
            tiebreaker = (triple, pair)
        elif flush:
            hand_rank = 6
            tiebreaker = sorted(ranks, reverse=True)
        elif straight:
            hand_rank = 5
            tiebreaker = straight_high
        elif 3 in counts:
            hand_rank = 4
            triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
            kickers = sorted([r for r in ranks if r != triple], reverse=True)
            tiebreaker = (triple, kickers)
        elif counts.count(2) >= 2:
            hand_rank = 3
            pairs = sorted([r for r, cnt in rank_counts.items() if cnt == 2], reverse=True)
            kicker = max(r for r in ranks if r not in pairs)
            tiebreaker = (pairs, kicker)
        elif 2 in counts:
            hand_rank = 2
            pair = max(r for r, cnt in rank_counts.items() if cnt == 2)
            kickers = sorted([r for r in ranks if r != pair], reverse=True)
            tiebreaker = (pair, kickers)
        else:
            hand_rank = 1
            tiebreaker = sorted(ranks, reverse=True)
        return (hand_rank, tiebreaker)

    # =============================================================================
    # Robust DQN Model with Dueling Architecture
    # =============================================================================
    class RobustDQN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(RobustDQN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.value_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values

    # =============================================================================
    # Opponent Policy Function
    # =============================================================================
    action_index_to_str = {i: action for i, action in enumerate(['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in'])}
    num_actions = len(action_index_to_str)

    def encode_obs(obs):
        # Encode the agent's hand as a one-hot vector (26 dims).
        card_to_index = {card: i for i, card in enumerate([r+s for s in ['H', 'S']
                                                            for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']])}
        hand_encoding = np.zeros(26)
        for card in obs['hand']:
            if card in card_to_index:
                hand_encoding[card_to_index[card]] = 1
        pot_value = np.array([np.log(obs['pot'] + 1.0)])
        # Encode beliefs: for each opponent (sorted by id), one-hot encode the cards that remain possible.
        belief_encoding = []
        for opp in sorted(obs['beliefs'].keys()):
            vec = np.zeros(26)
            possible = obs['beliefs'][opp]
            for card in possible:
                if card in card_to_index:
                    vec[card_to_index[card]] = 1
            belief_encoding.append(vec)
        belief_encoding = np.concatenate(belief_encoding) if belief_encoding else np.array([])
        state = np.concatenate([hand_encoding, pot_value, belief_encoding])
        return state.astype(np.float32)

    def make_opponent_policy(opponent_model):
        def policy_fn(obs):
            state = encode_obs(obs)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                q_values = opponent_model(state_tensor)
            action_idx = q_values.argmax().item()
            return action_index_to_str[action_idx]
        return policy_fn

    # =============================================================================
    # Epsilon Decay Function
    # =============================================================================
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 500

    def epsilon_by_frame(frame_idx):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

    # =============================================================================
    # Replay Buffer
    # =============================================================================
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch)
            return (np.array(state),
                    np.array(action),
                    np.array(reward, dtype=np.float32),
                    np.array(next_state),
                    np.array(done, dtype=np.float32))
        
        def __len__(self):
            return len(self.buffer)

    # =============================================================================
    # Training Loop
    # =============================================================================
    def train():
        num_episodes = 200000
        buffer_capacity = 10000
        batch_size = 32
        learning_rate = 1e-3
        gamma = 0.99
        target_update_freq = 100
        
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        env = FullPokerEnvWithOpponentPolicy()
        # Updated state dimension: 26 (hand) + 1 (pot) + (num_players-1)*26. For 6 players: 26+1+130 = 157.
        state_dim = 157
        agent = RobustDQN(input_dim=state_dim, output_dim=num_actions)
        target_net = RobustDQN(input_dim=state_dim, output_dim=num_actions)
        target_net.load_state_dict(agent.state_dict())
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        replay_buffer = ReplayBuffer(buffer_capacity)
        
        checkpoint_idx = 0
        global_agent_steps = 0
        episode_rewards = []
        metrics_list = []
        
        opponent_ids = [pid for pid in range(1, env.num_players)]
        opponent_update_idx = 0

        max_episode_steps = 500

        for episode in range(1, num_episodes + 1):
            obs = env.reset()
            state = encode_obs(obs)
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done and episode_steps < max_episode_steps:
                episode_steps += 1
                if env.current_player == env.agent_id:
                    epsilon = epsilon_by_frame(global_agent_steps)
                    if random.random() < epsilon:
                        action_idx = random.randrange(num_actions)
                    else:
                        state_tensor = torch.from_numpy(state).unsqueeze(0)
                        with torch.no_grad():
                            q_values = agent(state_tensor)
                        action_idx = q_values.argmax().item()
                    action_str = action_index_to_str[action_idx]
                    next_obs, reward, done, info = env.step(action_str)
                    next_state = encode_obs(next_obs)
                    replay_buffer.push(state, action_idx, reward, next_state, done)
                    state = next_state
                    global_agent_steps += 1
                    episode_reward += reward
                else:
                    _, reward, done, info = env.step('call')
                    episode_reward += reward

                if global_agent_steps % target_update_freq == 0 and global_agent_steps > 0:
                    target_net.load_state_dict(agent.state_dict())

                if len(replay_buffer) > batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    states_tensor = torch.from_numpy(states)
                    actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                    rewards_tensor = torch.from_numpy(rewards).unsqueeze(1)
                    next_states_tensor = torch.from_numpy(next_states)
                    dones_tensor = torch.from_numpy(dones).unsqueeze(1)

                    q_values = agent(states_tensor).gather(1, actions_tensor)
                    with torch.no_grad():
                        next_q_values = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
                    target = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
                    loss = F.mse_loss(q_values, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if episode_steps >= max_episode_steps and not done:
                if env.stage != "showdown":
                    obs, reward, done, info = env._finalize_hand()
                    episode_reward += reward
                    done = True

            episode_rewards.append(episode_reward)
            current_epsilon = epsilon_by_frame(global_agent_steps)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_reward
            metrics_list.append({
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'epsilon': current_epsilon
            })

            if episode % 50 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_idx}.pt")
                torch.save(agent.state_dict(), checkpoint_path)
                
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                random_checkpoint = random.choice(checkpoint_files)
                full_checkpoint_path = os.path.join(checkpoint_dir, random_checkpoint)
                
                opponent_to_update = opponent_ids[opponent_update_idx]
                opponent_model = RobustDQN(input_dim=state_dim, output_dim=num_actions)
                opponent_model.load_state_dict(torch.load(full_checkpoint_path))
                opponent_model.eval()
                env.opponent_policies[opponent_to_update] = make_opponent_policy(opponent_model)
                print(f"Updated opponent {opponent_to_update} using checkpoint {random_checkpoint}")
                
                opponent_update_idx = (opponent_update_idx + 1) % len(opponent_ids)
                checkpoint_idx = (checkpoint_idx + 1) % 100
            
            if episode % 100 == 0:
                print(f"Episode {episode} - Average bb/100: {avg_reward/25.0:.2f}, Epsilon: {current_epsilon:.2f}")
            
            if episode % 10000 == 0:
                with open("metrics.csv", "w", newline="") as csvfile:
                    fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metrics_list)
                print(f"Metrics saved to metrics.csv at episode {episode}")
        
        torch.save(agent.state_dict(), "final_agent_checkpoint.pt")
        with open("metrics.csv", "w", newline="") as csvfile:
            fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)
        print("Training complete. Metrics saved to metrics.csv")

    # =============================================================================
    # Main Execution
    # =============================================================================

    if __name__ == "__main__":
        # Optionally resume training from a checkpoint.
        checkpoint_path = "checkpoints/checkpoint_1"
        if os.path.exists(checkpoint_path):
            agent = RobustDQN(input_dim=53, output_dim=num_actions)
            agent.load_state_dict(torch.load(checkpoint_path))
            print("Resuming training from checkpoint.")
        train()
