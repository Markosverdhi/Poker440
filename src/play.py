from train import FullPokerEnvWithOpponentPolicy, RobustDQN
import torch
import numpy as np

def encode_obs(obs):
    """
    Encodes the observation into a fixed-length vector.
    - 26 dimensions for the agent’s hand (one-hot over 26 cards),
    - 1 dimension for the log-transformed pot,
    - 5 * 26 = 130 dimensions for the belief state for 5 opponents.
      If there are fewer than 5 opponents (e.g., heads-up play), the missing beliefs are padded with zeros.
    Total dimension = 26 + 1 + 130 = 157.
    """
    deck = [r+s for s in ['H', 'S'] for r in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']]
    card_to_index = {card: i for i, card in enumerate(deck)}
    
    hand_encoding = np.zeros(26)
    for card in obs['hand']:
        if card in card_to_index:
            hand_encoding[card_to_index[card]] = 1

    pot_value = np.array([np.log(obs['pot'] + 1.0)])
    
   
    belief_encoding_list = []
    num_opponents_expected = 5
    beliefs = obs.get('beliefs', {})
    for opp in range(1, num_opponents_expected+1):
        vec = np.zeros(26)
        if opp in beliefs:
            for card in beliefs[opp]:
                if card in card_to_index:
                    vec[card_to_index[card]] = 1
        belief_encoding_list.append(vec)
    belief_encoding = np.concatenate(belief_encoding_list)
    
    state = np.concatenate([hand_encoding, pot_value, belief_encoding])
    return state.astype(np.float32)

def my_make_opponent_policy(opponent_model):
    def policy_fn(obs):
        state = encode_obs(obs)
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            q_values = opponent_model(state_tensor)
        action_idx = q_values.argmax().item()
        action_index_to_str = {0: 'fold', 1: 'call', 2: 'check', 3: 'bet_small', 4: 'bet_big', 5: 'all_in'}
        return action_index_to_str[action_idx]
    return policy_fn

def play_game():
    env = FullPokerEnvWithOpponentPolicy()
    env.num_players = 2  
    env.stacks = {0: 1000, 1: 1000}
    
    checkpoint_path = "../Checkpoints/checkpoint_1.pt" 
    opponent_model = RobustDQN(input_dim=157, output_dim=6)
    opponent_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    opponent_model.eval()
    
    env.opponent_policies[1] = my_make_opponent_policy(opponent_model)
    
    print("Welcome to Heads-Up Poker with Belief States!")
    print("You are Player 0. Your opponent (Player 1) is controlled by a pre-trained model.\n")
    
    while True:
        obs = env.reset()
        print("\n--- New Hand ---")
        print(f"Dealer: Player {env.dealer}")
        print(f"Small Blind: Player {env.small_blind} (50 chips)")
        print(f"Big Blind: Player {env.big_blind} (100 chips)")
        print(f"Your hand: {obs['hand']}")
        print("Community Cards:", obs['community_cards'])
        print("Pot:", obs['pot'])
        print("Your Stack:", env.stacks[0])
        print("Opponent Stack:", env.stacks[1])
        print("Stage:", obs['stage'])
        print("Beliefs about opponents:", obs.get('beliefs'))
        
        done = False
        while not done:
            if env.current_player == 0:
                legal_actions = obs['legal_actions']
                if not legal_actions:
                    print("\nNo legal actions available. Finalizing hand...")
                    obs, reward, done, info = env._finalize_hand()
                    break
                print("\nYour turn.")
                print("Legal actions:", legal_actions)
                action = input("Enter your action: ").strip().lower()
                if action not in legal_actions:
                    print("Invalid action, defaulting to 'call'")
                    action = 'call'
                print("You chose:", action)
                obs, reward, done, info = env.step(action)
            else:
                print("\nOpponent's turn...")
                obs, reward, done, info = env.step('call') 
                print("Opponent has acted.")
            
            print("\n--- Updated Game State ---")
            print("Stage:", obs['stage'])
            print("Community Cards:", obs['community_cards'])
            print("Pot:", obs['pot'])
            print(f"Your Stack: {env.stacks[0]} | Your Bet: {env.current_bets[0]}")
            print(f"Opponent Stack: {env.stacks[1]} | Opponent Bet: {env.current_bets[1]}")
            if env.current_player is not None:
                print("Next to act: Player", env.current_player)
        
        print("\n--- Hand Over ---")
        print("Final Community Cards:", obs['community_cards'])
        print("Your hand:", env.hands[0])
        print("Opponent's hand:", env.hands[1])
        print("Winners:", info.get('winners', 'Unknown'))
        print("Hand Scores:", info.get('scores', 'N/A'))
        print("Net Reward:", reward)
        
        play_again = input("\nPlay another hand? (y/n): ").strip().lower()
        if play_again != 'y':
            break

if __name__ == "__main__":
    play_game()
