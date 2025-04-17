import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict # Import defaultdict
import os # Import os for path manipulation and directory creation


# --- Helper functions for parsing observations ---

def parse_observation(obs_str):
    """
    Given a JSON string from the ObservationDict column,
    parse it and return as a dictionary. Handles potential errors.
    """
    try:
        if isinstance(obs_str, dict): return obs_str
        if isinstance(obs_str, str):
             try: return json.loads(obs_str)
             except json.JSONDecodeError:
                  obs_str_fixed = obs_str.replace("'", '"').replace('None', 'null').replace('False', 'false').replace('True', 'true')
                  return json.loads(obs_str_fixed)
        else:
             print(f"Warning: Unexpected type for observation data: {type(obs_str)}. Returning empty dict.")
             return {}
    except Exception as e:
        return {} # Return empty dict on error

def extract_stage(obs):
    """ Extract the game stage from the observation dictionary. """
    return obs.get('stage', '').lower() if isinstance(obs, dict) else ''

def extract_hand(obs):
    """ Extract the player's hand from the observation dictionary. """
    return obs.get('hand', []) if isinstance(obs, dict) else []

# --- Hand classification and strength evaluation ---

def classify_hand(hand):
    """ Classify a two-card starting hand. """
    if not hand or len(hand) != 2: return "Unknown"
    def parse_card(card):
        if not isinstance(card, str) or len(card) < 2: return None, None
        rank = card[:-1]; suit = card[-1]
        if rank == '10': rank = 'T'
        return rank, suit
    rank1, suit1 = parse_card(hand[0]); rank2, suit2 = parse_card(hand[1])
    if rank1 is None or rank2 is None: return "Unknown"
    rank_order = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9, '8':8, '7':7, '6':6, '5':5, '4':4, '3':3, '2':2}
    if rank_order.get(rank1, 0) < rank_order.get(rank2, 0): rank1, rank2 = rank2, rank1
    if rank1 == rank2: return f"Pair {rank1}"
    else: suited_str = "Suited" if suit1 == suit2 else "Offsuit"; return f"{rank1}{rank2} {suited_str}"

def evaluate_hand_strength(hand_class):
    """ Assign a numerical strength to the starting hand based on its classification. """
    card_rank_map = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9, '8':8, '7':7, '6':6, '5':5, '4':4, '3':3, '2':2}
    if hand_class.startswith("Pair"):
        try: rank = hand_class.split()[1]; return card_rank_map.get(rank, 0) + 50 if rank in card_rank_map else 0
        except Exception: return 0
    elif " " in hand_class:
        try:
            parts = hand_class.split(); cards = parts[0]; suited_text = parts[1].lower()
            if len(parts) != 2 or len(cards) != 2 or cards[0] not in card_rank_map or cards[1] not in card_rank_map: return 0
            val1 = card_rank_map.get(cards[0], 0); val2 = card_rank_map.get(cards[1], 0)
            score = val1 + val2 + (5 if suited_text == "suited" else 0)
            return score
        except Exception: return 0
    else: return 0

# --- Analysis functions ---

# *** MODIFIED: Added agent_player_id argument ***
def analyze_decisions(df, agent_player_id=1):
    """
    Analyze decision frequencies, rewards, and hand strength for the agent
    during the preflop stage. Returns summaries as DataFrames.
    Uses the provided agent_player_id.
    """
    # Filter for agent actions and preflop stage.
    if 'PlayerID' not in df.columns or 'Stage' not in df.columns:
         print("Error: Missing 'PlayerID' or 'Stage' column in the log file.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0
    # *** Use the passed agent_player_id ***
    df_agent = df[(df['PlayerID'] == agent_player_id) & (df['Stage'] == 'preflop')].copy()

    if df_agent.empty:
        print(f"No preflop actions found for agent (PlayerID {agent_player_id}).")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0

    if 'Action' not in df_agent.columns or 'Reward' not in df_agent.columns:
         print("Error: Missing 'Action' or 'Reward' column in the log file.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0

    action_counts = df_agent['Action'].value_counts().rename_axis('Action').reset_index(name='Count')
    df_agent['Reward'] = pd.to_numeric(df_agent['Reward'], errors='coerce')
    df_agent.dropna(subset=['Reward'], inplace=True)
    action_rewards = df_agent.groupby('Action')['Reward'].mean().reset_index(name='Avg_Reward')
    action_summary = pd.merge(action_counts, action_rewards, on='Action', how='left')

    if 'Hand' not in df_agent.columns:
         print("Error: Missing 'Hand' column in the log file.")
         return action_summary, pd.DataFrame(), pd.DataFrame(), 0.0

    df_agent['Hand_Class'] = df_agent['Hand'].apply(classify_hand)
    df_agent['Hand_Strength'] = df_agent['Hand_Class'].apply(evaluate_hand_strength)
    df_agent_valid_hands = df_agent[df_agent['Hand_Class'] != "Unknown"].copy()

    if df_agent_valid_hands.empty:
        print("No valid hands found for analysis after classification.")
        return action_summary, pd.DataFrame(), pd.DataFrame(), 0.0

    hand_distribution = df_agent_valid_hands['Hand_Class'].value_counts().rename_axis('Hand_Class').reset_index(name='Count')
    aggressive_actions = ['bet_small', 'bet_big', 'all_in']; passive_actions = ['fold', 'call', 'check']
    df_agent_valid_hands['Action_Type'] = df_agent_valid_hands['Action'].apply(lambda x: 'Aggressive' if x in aggressive_actions else ('Passive' if x in passive_actions else 'Other'))
    strength_by_action = df_agent_valid_hands[df_agent_valid_hands['Action_Type'].isin(['Aggressive', 'Passive'])].groupby('Action_Type')['Hand_Strength'].mean().reset_index(name='Mean_Hand_Strength')

    correlation = 0.0
    if pd.api.types.is_numeric_dtype(df_agent_valid_hands['Hand_Strength']) and pd.api.types.is_numeric_dtype(df_agent_valid_hands['Reward']) and len(df_agent_valid_hands) > 1:
        correlation = df_agent_valid_hands['Hand_Strength'].corr(df_agent_valid_hands['Reward'])
        if pd.isna(correlation): correlation = 0.0

    return action_summary, hand_distribution, strength_by_action, correlation

# --- Plotting functions (paths updated previously) ---

def plot_action_distribution(action_summary, output_dir):
    """ Plot action frequency. """
    if action_summary.empty: print("Skipping action distribution plot: No data."); return
    plt.figure(figsize=(8, 6)); plt.bar(action_summary['Action'], action_summary['Count'], color='skyblue')
    plt.xlabel("Action"); plt.ylabel("Frequency"); plt.title("Agent Action Frequency (Preflop Decisions)")
    plt.tight_layout(); plot_path = os.path.join("plots", "action_distribution.png")
    plt.savefig(plot_path); plt.close(); print(f"Saved action distribution plot to {plot_path}")

def plot_hand_class_distribution(hand_counts, output_dir):
    """ Plot hand class distribution. """
    if hand_counts.empty: print("Skipping hand distribution plot: No data."); return
    top_n = 30; hand_counts_plot = hand_counts.head(top_n)
    plt.figure(figsize=(12, 7)); plt.bar(hand_counts_plot['Hand_Class'], hand_counts_plot['Count'], color='lightgreen')
    plt.xlabel("Hand Classification"); plt.ylabel("Frequency"); plt.title(f"Distribution of Agent Starting Hands (Top {top_n}, Preflop)")
    plt.xticks(rotation=75); plt.tight_layout(); plot_path = os.path.join("plots", "hand_distribution.png")
    plt.savefig(plot_path); plt.close(); print(f"Saved hand distribution plot to {plot_path}")

def plot_strength_by_action(strength_df, output_dir):
    """ Plot mean hand strength by action type. """
    if strength_df.empty: print("Skipping strength by action plot: No data."); return
    plt.figure(figsize=(6, 4)); plt.bar(strength_df['Action_Type'], strength_df['Mean_Hand_Strength'], color='salmon')
    plt.xlabel("Action Type"); plt.ylabel("Mean Hand Strength"); plt.title("Mean Hand Strength by Action Type")
    plt.tight_layout(); plot_path = os.path.join("plots", "strength_by_action.png")
    plt.savefig(plot_path); plt.close(); print(f"Saved strength by action plot to {plot_path}")

# --- Best practice comparison (unchanged) ---
def compare_with_best_practices(strength_by_action, correlation):
    commentary = "\nComparison to Best Poker Practices:\n"
    if strength_by_action.empty or len(strength_by_action) < 2:
        commentary += " - Insufficient data to compare aggressive vs. passive hand strength.\n"
    else:
        try:
            agg_strength_series = strength_by_action[strength_by_action['Action_Type'] == 'Aggressive']['Mean_Hand_Strength']
            pas_strength_series = strength_by_action[strength_by_action['Action_Type'] == 'Passive']['Mean_Hand_Strength']
            agg_strength = agg_strength_series.iloc[0] if not agg_strength_series.empty else None
            pas_strength = pas_strength_series.iloc[0] if not pas_strength_series.empty else None
            if agg_strength is not None: commentary += f" - Mean Hand Strength when playing Aggressively: {agg_strength:.2f}\n"
            else: commentary += " - No aggressive actions found.\n"
            if pas_strength is not None: commentary += f" - Mean Hand Strength when playing Passively: {pas_strength:.2f}\n"
            else: commentary += " - No passive actions found.\n"
            if agg_strength is not None and pas_strength is not None:
                commentary += "   (As expected, aggressive actions used with stronger hands.)\n" if agg_strength > pas_strength else "   (Unexpected: aggressive actions do not have higher average strength.)\n"
            elif agg_strength is None or pas_strength is None: commentary += "   (Cannot compare due to missing data.)\n"
        except Exception as e: commentary += f" - Error processing strength data: {e}\n"
    commentary += f" - Correlation between Hand Strength and Reward: {correlation:.2f}\n"
    if correlation > 0.3: commentary += "   (Positive correlation suggests better hands yield higher rewards.)\n"
    elif correlation < -0.1: commentary += "   (Negative correlation is unexpected, warrants investigation.)\n"
    else: commentary += "   (Weak correlation suggests decisions may not strongly align with hand strength.)\n"
    commentary += "\nStandard preflop guidelines advise aggression with premium holdings. Use results for tuning.\n"
    return commentary

# --- Main analysis function ---

def main():
    parser = argparse.ArgumentParser(description="Decision Analysis for Poker RL Agent Simulation")
    parser.add_argument("--detailed_log", type=str, default="Output_CSVs/detailed_simulation_log.csv", help="Path to the detailed simulation log CSV file.")
    parser.add_argument("--output_analysis", type=str, default="Output_CSVs/decision_analysis_summary.csv", help="Path to save the analysis summary CSV.")
    parser.add_argument("--output_dir", type=str, default="Output_CSVs", help="Directory to save output plots and summary file.")
    # *** ADDED agent_id argument definition ***
    parser.add_argument("--agent_id", type=int, default=1, help="Player ID (1-based) of the agent to analyze.")
    args = parser.parse_args()

    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)
    print(f"Ensured output directory exists: {output_directory}")

    detailed_log_path = args.detailed_log
    output_analysis_path = os.path.join(output_directory, os.path.basename(args.output_analysis))

    try:
        df = pd.read_csv(detailed_log_path)
        print(f"Log loaded successfully from {detailed_log_path}. Rows: {len(df)}")
    except FileNotFoundError: print(f"Error: Log file not found at {detailed_log_path}"); return
    except Exception as e: print(f"Error reading log file {detailed_log_path}: {e}"); return

    obs_col_name = 'ObservationDict'
    if obs_col_name not in df.columns: print(f"Error: Column '{obs_col_name}' not found."); return

    df['ObservationDictParsed'] = df[obs_col_name].apply(parse_observation)
    df['Stage'] = df['ObservationDictParsed'].apply(extract_stage)
    df['Hand'] = df['ObservationDictParsed'].apply(extract_hand)

    # *** Pass args.agent_id to the analysis function ***
    action_summary, hand_distribution, strength_by_action, strength_reward_corr = analyze_decisions(df, agent_player_id=args.agent_id)

    # Save summary analysis to CSV.
    try:
        with open(output_analysis_path, "w") as f:
            f.write(f"Analysis for Agent ID: {args.agent_id}\n\n") # Add Agent ID to report
            f.write("Action Summary (Preflop Decisions):\n")
            f.write(action_summary.to_csv(index=False, lineterminator='\n') if not action_summary.empty else "No action data.\n")
            f.write("\n\nStarting Hand Distribution (Preflop):\n")
            f.write(hand_distribution.to_csv(index=False, lineterminator='\n') if not hand_distribution.empty else "No hand data.\n")
            f.write("\n\nMean Hand Strength by Action Type:\n")
            f.write(strength_by_action.to_csv(index=False, lineterminator='\n') if not strength_by_action.empty else "No strength data.\n")
            f.write("\n\nCorrelation between Hand Strength and Reward:\n")
            f.write(f"{strength_reward_corr:.2f}\n")
            commentary = compare_with_best_practices(strength_by_action, strength_reward_corr)
            f.write("\n" + commentary)
        print("Analysis summary saved to", output_analysis_path)
    except Exception as e: print(f"Error writing analysis summary: {e}")

    # Print summaries to console
    print(f"\n--- Analysis Summary for Agent ID: {args.agent_id} ---")
    print("\n--- Action Summary (Preflop) ---")
    print(action_summary.to_string(index=False) if not action_summary.empty else "No data.")
    print("\n--- Starting Hand Distribution (Preflop) ---")
    print(hand_distribution.to_string(index=False) if not hand_distribution.empty else "No data.")
    print("\n--- Mean Hand Strength by Action Type ---")
    print(strength_by_action.to_string(index=False) if not strength_by_action.empty else "No data.")
    print("\n--- Correlation between Hand Strength and Reward ---")
    print(f"{strength_reward_corr:.2f}")
    commentary = compare_with_best_practices(strength_by_action, strength_reward_corr)
    print(commentary)

    # Plot visualizations.
    plot_action_distribution(action_summary, output_directory)
    plot_hand_class_distribution(hand_distribution, output_directory)
    plot_strength_by_action(strength_by_action, output_directory)

if __name__ == "__main__":
    main()
