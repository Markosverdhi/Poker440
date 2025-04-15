import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Helper functions for parsing observations ---

def parse_observation(obs_str):
    """
    Given a JSON string from the Observation column,
    parse it and return as a dictionary.
    """
    try:
        obs = json.loads(obs_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        obs = {}
    return obs

def extract_stage(obs):
    """
    Extract the game stage from the observation.
    Default to empty string if not available.
    """
    return obs.get('stage', '').lower() if isinstance(obs, dict) else ''

def extract_hand(obs):
    """
    Extract the player's hand from the observation.
    Expected to be a list like ['AH', 'KD'].
    """
    return obs.get('hand', []) if isinstance(obs, dict) else []

# --- Hand classification and strength evaluation ---

def classify_hand(hand):
    """
    Classify a two-card starting hand.
    Returns a string such as:
      - "Pair AA" if both cards are the same rank,
      - "AK Suited" if cards are of different rank and same suit,
      - "AK Offsuit" if cards are of different rank and different suit.
    Assumes hand is a list like ['AH', 'KD'].
    """
    if not hand or len(hand) != 2:
        return "Unknown"
    
    def parse_card(card):
        # Supports cards like '10H' or 'AH'
        if len(card) == 3:
            return card[:2], card[2]
        else:
            return card[0], card[1]
    
    rank1, suit1 = parse_card(hand[0])
    rank2, suit2 = parse_card(hand[1])
    
    if rank1 == rank2:
        return f"Pair {rank1}"
    else:
        # Order higher rank first.
        # For simplicity, use a ranking order (A high, 2 low)
        rank_order = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9, '8':8, '7':7,
                      '6':6, '5':5, '4':4, '3':3, '2':2}
        if rank_order.get(rank1, 0) < rank_order.get(rank2, 0):
            rank1, rank2 = rank2, rank1
        suited_str = "Suited" if suit1 == suit2 else "Offsuit"
        return f"{rank1}{rank2} {suited_str}"

def evaluate_hand_strength(hand_class):
    """
    Assign a numerical strength to the starting hand based on its classification.
    This is a simple heuristic:
      - Pairs get a high bonus plus the rank value.
      - For non-pairs, the sum of card values is used with a bonus if suited.
    The ranking order is: A=14, K=13, Q=12, J=11, T=10, ..., 2=2.
    """
    card_rank_map = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9,
                     '8':8, '7':7, '6':6, '5':5, '4':4, '3':3, '2':2}
    
    if hand_class.startswith("Pair"):
        try:
            rank = hand_class.split()[1]
            return card_rank_map.get(rank, 0) + 50  # bonus for pair
        except Exception:
            return 0
    else:
        try:
            parts = hand_class.split()
            if len(parts) != 2:
                return 0
            cards = parts[0]  # e.g., "AK"
            suited_text = parts[1].lower()
            if len(cards) != 2:
                return 0
            val1 = card_rank_map.get(cards[0], 0)
            val2 = card_rank_map.get(cards[1], 0)
            score = val1 + val2
            if suited_text == "suited":
                score += 5  # bonus for suited
            return score
        except Exception:
            return 0

# --- Analysis functions ---

def analyze_decisions(df):
    """
    Analyze decision frequencies, rewards, and hand strength for the agent (seat 0)
    during the preflop stage. Returns summaries as DataFrames.
    """
    # Filter for agent actions (seat 0) and preflop stage.
    df_agent = df[(df['Current_Player'] == 0) & (df['Stage'] == 'preflop')]
    
    # Count action frequencies.
    action_counts = df_agent['Action'].value_counts().rename_axis('Action').reset_index(name='Count')
    
    # Average reward per action.
    action_rewards = df_agent.groupby('Action')['Reward'].mean().reset_index(name='Avg_Reward')
    action_summary = pd.merge(action_counts, action_rewards, on='Action')
    
    # Classify hands and compute hand strength.
    df_agent['Hand_Class'] = df_agent['Hand'].apply(classify_hand)
    df_agent['Hand_Strength'] = df_agent['Hand_Class'].apply(evaluate_hand_strength)
    
    hand_distribution = df_agent['Hand_Class'].value_counts().rename_axis('Hand_Class').reset_index(name='Count')
    
    # Divide actions into aggressive and passive.
    aggressive_actions = ['bet_small', 'bet_big', 'all_in']
    passive_actions = ['fold', 'call', 'check']
    df_agent['Action_Type'] = df_agent['Action'].apply(lambda x: 'Aggressive' if x in aggressive_actions else 'Passive')
    
    strength_by_action = df_agent.groupby('Action_Type')['Hand_Strength'].mean().reset_index(name='Mean_Hand_Strength')
    
    # Also compute correlation between hand strength and reward.
    correlation = df_agent['Hand_Strength'].corr(df_agent['Reward'])
    
    return action_summary, hand_distribution, strength_by_action, correlation

def plot_action_distribution(action_summary):
    """
    Plot a bar chart of the frequency of each action taken by the agent.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(action_summary['Action'], action_summary['Count'], color='skyblue')
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Agent Action Frequency (Preflop Decisions)")
    plt.tight_layout()
    plt.savefig("plots/action_distribution.png")
    plt.close()

def plot_hand_class_distribution(hand_counts):
    """
    Plot a bar chart of the distribution of starting hand classifications.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(hand_counts['Hand_Class'], hand_counts['Count'], color='lightgreen')
    plt.xlabel("Hand Classification")
    plt.ylabel("Frequency")
    plt.title("Distribution of Agent Starting Hands (Preflop)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/hand_distribution.png")
    plt.close()

def plot_strength_by_action(strength_df):
    """
    Plot a bar chart of the average hand strength by action type (Aggressive vs Passive).
    """
    plt.figure(figsize=(6, 4))
    plt.bar(strength_df['Action_Type'], strength_df['Mean_Hand_Strength'], color='salmon')
    plt.xlabel("Action Type")
    plt.ylabel("Mean Hand Strength")
    plt.title("Mean Hand Strength by Action Type")
    plt.tight_layout()
    plt.savefig("plots/strength_by_action.png")
    plt.close()

def compare_with_best_practices(strength_by_action, correlation):
    """
    Generate a comparison commentary based on best poker practices.
    
    Best practices (as per the full-preflop-charts :contentReference[oaicite:1]{index=1} and other literature)
    indicate that:
      - Aggressive actions (raises, 3-bets, all in) should be taken primarily with hands
        that have higher intrinsic strength (e.g., high pairs, strong suited connectors like AKs, AQs).
      - Passive actions (calls, checks, folds) are more common with marginal or weak holdings.
      
    This function compares the mean hand strength for aggressive versus passive actions
    and reports the correlation between hand strength and rewards (as a proxy for win probability
    estimation). A strong positive correlation and a significantly higher mean for aggressive actions
    would indicate alignment with best practices.
    """
    agg_strength = strength_by_action[strength_by_action['Action_Type'] == 'Aggressive']['Mean_Hand_Strength'].iloc[0]
    pas_strength = strength_by_action[strength_by_action['Action_Type'] == 'Passive']['Mean_Hand_Strength'].iloc[0]
    
    commentary = "\nComparison to Best Poker Practices:\n"
    commentary += f" - Mean Hand Strength when playing Aggressively: {agg_strength:.2f}\n"
    commentary += f" - Mean Hand Strength when playing Passively: {pas_strength:.2f}\n"
    if agg_strength > pas_strength:
        commentary += "   (As expected, the agent tends to take aggressive actions with stronger hands.)\n"
    else:
        commentary += "   (Unexpected: the agent's aggressive actions do not appear to have a higher average hand strength.)\n"
    
    commentary += f" - Correlation between Hand Strength and Reward: {correlation:.2f}\n"
    if correlation > 0.3:
        commentary += "   (A positive correlation suggests that better hands are generally resulting in higher rewards, which aligns with fundamental poker strategy.)\n"
    else:
        commentary += "   (The weak correlation may indicate that the agent's decisions are not strongly aligned with hand strength, suggesting further review is needed.)\n"
    
    commentary += ("\nAccording to standard preflop guidelines (e.g., those in the full-preflop-charts), one should "
                     "generally be more aggressive with premium holdings such as high pairs (AA, KK, etc.) and strong "
                     "suited connectors (AKs, AQs) and more passive with marginal hands. Use these results to guide further "
                     "tuning of the agent's decision policy.\n")
    return commentary

# --- Main analysis function ---

def main():
    parser = argparse.ArgumentParser(description="Decision Analysis for Poker RL Agent Simulation")
    parser.add_argument("--detailed_log", type=str, default="detailed_simulation_log.csv",
                        help="Path to the detailed simulation log CSV file.")
    parser.add_argument("--output_analysis", type=str, default="decision_analysis_summary.csv",
                        help="Path to save the analysis summary.")
    args = parser.parse_args()

    # Read the detailed simulation log
    # Expected CSV header: Episode, Step, Current_Player, Action, Reward, Observation
    df = pd.read_csv(args.detailed_log)
    
    # Parse the Observation JSON into Stage and Hand columns.
    df['Stage'] = df['Observation'].apply(lambda x: extract_stage(parse_observation(x)))
    df['Hand'] = df['Observation'].apply(lambda x: extract_hand(parse_observation(x)))
    
    # Analyze decisions for agent (seat 0) during preflop.
    action_summary, hand_distribution, strength_by_action, strength_reward_corr = analyze_decisions(df)
    
    # Save summary analysis to CSV.
    with open(args.output_analysis, "w") as f:
        f.write("Action Summary (Preflop Decisions):\n")
        action_summary.to_csv(f, index=False)
        f.write("\n\nStarting Hand Distribution (Preflop):\n")
        hand_distribution.to_csv(f, index=False)
        f.write("\n\nMean Hand Strength by Action Type:\n")
        strength_by_action.to_csv(f, index=False)
        f.write("\n\nCorrelation between Hand Strength and Reward:\n")
        f.write(f"{strength_reward_corr:.2f}\n")
        
        # Append best-practice comparison commentary.
        commentary = compare_with_best_practices(strength_by_action, strength_reward_corr)
        f.write("\n" + commentary)
    
    print("Analysis summary saved to", args.output_analysis)
    print("\n--- Action Summary (Preflop) ---")
    print(action_summary)
    print("\n--- Starting Hand Distribution (Preflop) ---")
    print(hand_distribution)
    print("\n--- Mean Hand Strength by Action Type ---")
    print(strength_by_action)
    print("\n--- Correlation between Hand Strength and Reward ---")
    print(f"{strength_reward_corr:.2f}")
    
    # Print commentary on best-practice comparison.
    commentary = compare_with_best_practices(strength_by_action, strength_reward_corr)
    print(commentary)
    
    # Plot visualizations.
    plot_action_distribution(action_summary)
    plot_hand_class_distribution(hand_distribution)
    plot_strength_by_action(strength_by_action)

if __name__ == "__main__":
    main()
