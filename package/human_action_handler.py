# decision_analysis_v3.py (Expanded Charts)
import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Hand Classification Logic (Unchanged) ---

RANKS_ORDER = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_MAP = {r: i for i, r in enumerate(RANKS_ORDER)}

def classify_hand(hand):
    """ Classifies a two-card starting hand. """
    if not isinstance(hand, list) or len(hand) != 2: return "Unknown"
    try:
        card1, card2 = hand
        rank1 = card1[:-1] if len(card1) > 2 else card1[0]; suit1 = card1[-1]
        rank2 = card2[:-1] if len(card2) > 2 else card2[0]; suit2 = card2[-1]
        if rank1 == '10': rank1 = 'T'
        if rank2 == '10': rank2 = 'T'
        if rank1 not in RANK_MAP or rank2 not in RANK_MAP: return "Unknown"
        if RANK_MAP[rank1] < RANK_MAP[rank2]: rank1, rank2 = rank2, rank1
        if rank1 == rank2: return f"{rank1}{rank2}"
        elif suit1 == suit2: return f"{rank1}{rank2}s"
        else: return f"{rank1}{rank2}o"
    except Exception: return "Unknown"

# --- Preflop Chart Data (Manual Encoding - Expanded Subset) ---

# Encoding actions: R = Raise (RFI), F = Fold, L = Limp (SB RFI),
#                   3B = 3Bet (Facing RFI), C = Call (Facing RFI)
# Based on charts from full-preflop-charts.pdf
# Assumes 100bb stacks, ante in play.

def get_all_hand_classes():
    # (Function unchanged)
    classes = set();
    for i in range(len(RANKS_ORDER)):
        for j in range(i, len(RANKS_ORDER)):
            r1 = RANKS_ORDER[j]; r2 = RANKS_ORDER[i]
            if i == j: classes.add(f"{r1}{r2}")
            else: classes.add(f"{r1}{r2}s"); classes.add(f"{r1}{r2}o")
    return list(classes)

ALL_HAND_CLASSES = get_all_hand_classes()

# --- Initialize Chart Data Structure ---
# Structure: CHARTS[situation_type][player_pos][opponent_pos (if applicable)][hand_class] = action
PREFLOP_CHARTS = {
    'RFI': { # Raise First In situation
        pos: {hand: 'F' for hand in ALL_HAND_CLASSES}
        for pos in ['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN', 'SB']
    },
    'Facing_RFI': { # Facing a Raise First In situation
        player_pos: {
            opp_pos: {hand: 'F' for hand in ALL_HAND_CLASSES}
            for opp_pos in ['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN', 'SB'] # Potential raiser positions
        }
        for player_pos in ['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB'] # Player positions
    }
    # Add 'Vs_3Bet' structure here if encoding those charts later
}

# --- Populate RFI Charts (Page 3) ---
# (Code for populating RFI charts remains the same as in v2)
utg_raises = ['77','88','99','TT','JJ','QQ','KK','AA', 'AQs','AJs','ATs','A9s','A8s','A7s','A6s','A5s', 'AKo','AQo', 'KQs','KJs','KTs','K9s', 'QJs','QTs', 'JTs', 'T9s'];
for h in utg_raises: PREFLOP_CHARTS['RFI']['UTG'][h] = 'R'
utg1_raises = utg_raises + ['66','A4s','A3s','AJo','KJo','KQo','QJo','J9s','T8s','98s','87s'];
for h in utg1_raises: PREFLOP_CHARTS['RFI']['UTG+1'][h] = 'R'
utg2_raises = utg1_raises + ['55','A2s','ATo','KTo','QTo','JTo','Q9s','T9s','98s','87s','76s','65s'];
for h in utg2_raises: PREFLOP_CHARTS['RFI']['UTG+2'][h] = 'R'; PREFLOP_CHARTS['RFI']['LJ'][h] = 'R'
hj_raises = utg2_raises + ['44','33','22','A9o','K9o','Q9o','J9o','T9o','97s','76s','65s','54s'];
for h in hj_raises: PREFLOP_CHARTS['RFI']['HJ'][h] = 'R'
co_raises = hj_raises + ['A8o','A7o','A6o','A5o','A4o','A3o','A2o', 'K8o','K7o','K6o','K5o', 'Q8o', 'J8o', 'T8o', '98o', 'K2s','Q2s','J7s','J6s','J5s','J4s','J3s','J2s', 'T7s','T6s', '96s','95s', '86s','85s', '75s', '64s', '53s','52s', '43s','42s', '32s'];
for h in co_raises: PREFLOP_CHARTS['RFI']['CO'][h] = 'R'
btn_raises = co_raises + ['K4o','K3o','K2o', 'Q7o','Q6o','Q5o','Q4o','Q3o','Q2o', 'J7o','J6o','J5o','J4o','J3o','J2o', 'T7o','T6o','T5o','T4o','T3o','T2o', '97o','96o','95o','94o','93o','92o', '87o','86o','85o','84o','83o','82o', '76o','75o','74o', '65o','64o','63o','62o', '54o','53o','52o', '43o','42o', '32o'];
for h in btn_raises: PREFLOP_CHARTS['RFI']['BTN'][h] = 'R'
sb_value_raises = ['88','99','TT','JJ','QQ','KK','AA', 'AQs','AJs','ATs','A9s','A8s','A7s', 'AKo','AQo','AJo', 'KQs','KJs','KTs','K9s', 'KQo','KJo', 'QJs','QTs','Q9s', 'QJo', 'JTs','J9s', 'JTo', 'T9s','T8s', 'T9o'];
sb_bluff_raises = ['A6s','A5s','A4s','A3s','A2s', 'A8o','A7o','A6o','A5o','A4o','A3o','A2o', 'K8s','K7s','K6s','K5s','K4s','K3s','K2s', 'K8o','K7o','K6o','K5o','K4o','K3o','K2o', 'Q8s','Q7s','Q6s','Q5s','Q4s','Q3s','Q2s', 'Q8o','Q7o','Q6o','Q5o','Q4o','Q3o','Q2o', 'J8s','J7s','J6s','J5s','J4s','J3s','J2s', 'J8o','J7o','J6o','J5o','J4o','J3o','J2o', 'T7s','T6s','T5s','T4s','T3s','T2s', 'T8o','T7o','T6o','T5o','T4o','T3o','T2o', '98s','97s','96s','95s','94s','93s','92s', '98o','97o','96o','95o','94o','93o','92o', '87s','86s','85s','84s','83s','82s', '87o','86o','85o','84o','83o','82o', '76s','75s','74s','73s','72s', '76o','75o','74o','73o','72o', '65s','64s','63s','62s', '65o','64o','63o','62o', '54s','53s','52s', '54o','53o','52o', '44','33','22', '43s','42s', '43o','42o', '32s', '32o'];
sb_limps = ['77','66','55', 'A9o', 'K9o', 'Q9o', 'J9o', 'T9o'];
for h in sb_value_raises: PREFLOP_CHARTS['RFI']['SB'][h] = 'R'
for h in sb_bluff_raises: PREFLOP_CHARTS['RFI']['SB'][h] = 'R' # Treat bluffs as 'Raise'
for h in sb_limps: PREFLOP_CHARTS['RFI']['SB'][h] = 'L' # Mark limps

# --- Populate Facing RFI Charts (Subset - Page 5 & 8) ---
# Actions: 3B = 3Bet (Value or Bluff), C = Call, F = Fold

# BTN vs UTG RFI (Page 5, Top Left Chart)
btn_vs_utg_3bet = ['JJ','QQ','KK','AA', 'AKs','AQs','AJs','ATs', 'AKo','AQo', 'KQs', 'A5s','A4s', 'K9s', 'Q9s', 'J9s', 'T8s', '97s', '86s', '75s', '64s', '53s'] # Value + Bluff examples
btn_vs_utg_call = ['TT','99','88','77', 'A9s','A8s','A7s','A6s', 'AJo', 'KTs', 'KJo', 'QTs','QJs', 'QJo', 'JTs', 'T9s', '98s', '87s', '76s', '65s', '54s']
for h in btn_vs_utg_3bet: PREFLOP_CHARTS['Facing_RFI']['BTN']['UTG'][h] = '3B'
for h in btn_vs_utg_call: PREFLOP_CHARTS['Facing_RFI']['BTN']['UTG'][h] = 'C'
# All others default to 'F' (Fold)

# BB vs BTN RFI (Page 8, Bottom Middle Chart)
bb_vs_btn_3bet = ['99','TT','JJ','QQ','KK','AA', 'AQs','AJs','ATs','A9s','A8s','A7s','A6s','A5s', 'AKo','AQo','AJo','ATo','A9o','A8o','A7o','A6o','A5o', 'KQs','KJs','KTs','K9s','K8s', 'KQo','KJo','KTo','K9o', 'QJs','QTs','Q9s', 'QJo','QTo', 'JTs','J9s', 'JTo', 'T9s','T8s', 'T9o', '98s', 'K7s','K6s','K5s','K4s','K3s','K2s', 'Q8s','Q7s','Q6s','Q5s','Q4s','Q3s','Q2s', 'J8s','J7s', 'T7s', '97s','96s', '87s','86s','85s', '76s','75s','74s', '65s','64s', '54s','53s', '43s'] # Value + Bluff examples
bb_vs_btn_call = ['22','33','44','55','66','77','88', 'A4s','A3s','A2s', 'A4o','A3o','A2o', 'K8o','K7o','K6o','K5o','K4o','K3o','K2o', 'Q9o','Q8o','Q7o','Q6o','Q5o','Q4o','Q3o','Q2o', 'J9o','J8o','J7o','J6o','J5o','J4o','J3o','J2o', 'T8o','T7o','T6o','T5o','T4o','T3o','T2o', '98o','97o','96o','95o','94o','93o','92o', '87o','86o','85o','84o','83o','82o', '76o','75o','74o','73o','72o', '65o','64o','63o','62o', '54o','53o','52o', '43o','42o', '32o', 'J6s','J5s','J4s','J3s','J2s', 'T6s','T5s','T4s','T3s','T2s', '95s','94s','93s','92s', '84s','83s','82s', '73s','72s', '63s','62s', '52s', '42s', '32s']
for h in bb_vs_btn_3bet: PREFLOP_CHARTS['Facing_RFI']['BB']['BTN'][h] = '3B'
for h in bb_vs_btn_call: PREFLOP_CHARTS['Facing_RFI']['BB']['BTN'][h] = 'C'
# All others default to 'F' (Fold)

# --- End Chart Encoding ---

def get_chart_action(situation_type, player_pos, opp_pos, hand_class):
    """Looks up the recommended action from the encoded charts."""
    if situation_type == 'RFI':
        # Handle RFI position mapping if needed
        if player_pos not in PREFLOP_CHARTS['RFI']:
            if player_pos == 'UTG+1': player_pos = 'UTG'
            elif player_pos == 'UTG+2': player_pos = 'LJ'
            elif player_pos == 'MP': player_pos = 'LJ'
            else: return "Unknown Position"
        chart = PREFLOP_CHARTS['RFI'].get(player_pos)
        return chart.get(hand_class, 'F') if chart else "Unknown Position" # Default RFI to Fold

    elif situation_type == 'Facing_RFI':
        if player_pos not in PREFLOP_CHARTS['Facing_RFI'] or \
           opp_pos not in PREFLOP_CHARTS['Facing_RFI'][player_pos]:
            return "Unknown Situation" # Specific player/opponent combo not encoded
        chart = PREFLOP_CHARTS['Facing_RFI'][player_pos].get(opp_pos)
        return chart.get(hand_class, 'F') if chart else "Unknown Situation" # Default Facing RFI to Fold

    # Add other situation types like 'Vs_3Bet' here later
    else:
        return "Unknown Situation Type"


# --- Analysis Functions ---

def parse_observation(obs_str):
    """ Safely parses the JSON observation string. """
    try:
        if isinstance(obs_str, str):
            obs_str = obs_str.replace("'", '"').replace('None', 'null')
        return json.loads(obs_str)
    except Exception: return {}

def determine_situation(row, agent_player_id):
    """
    Tries to determine the preflop situation (RFI, Facing RFI).
    NOTE: This requires accurate logging of prior actions or more context.
    """
    if row['Stage'] != 'preflop':
        return None, None # Not preflop

    # Use logged RFI flag if available
    if row.get('IsRFIOpportunity', False):
        return 'RFI', None # RFI situation, no specific opponent

    # --- Attempt to identify Facing RFI ---
    # This part is highly dependent on having more log context.
    # Placeholder logic: Check if current_bet > BB and agent hasn't acted yet.
    # Needs info about who made the current_bet (last aggressor).
    # TODO: Enhance logging in simulate.py to include last aggressor info.
    obs_dict = row['ObservationDict']
    current_bet = obs_dict.get('current_bet', 0)
    player_current_bet = obs_dict.get('player_current_bet', 0)
    # Assuming BB=100 for simplicity
    BIG_BLIND = 100
    if current_bet > BIG_BLIND and player_current_bet <= BIG_BLIND:
        # This *might* be facing an RFI, but we don't know who raised.
        # Return a generic 'Facing_RFI' and None for opponent position.
        # The analysis will likely fail chart lookup without opponent pos.
        return 'Facing_RFI', 'Unknown' # Mark opponent as Unknown
    else:
        # Could be other situations (limped pot, facing 3bet, etc.)
        return None, None


def analyze_preflop_vs_charts(df, agent_player_id=1):
    """
    Analyzes Agent's Preflop decisions against encoded charts.
    Handles RFI and attempts to handle Facing RFI based on available logs.
    """
    df_agent_preflop = df[(df['PlayerID'] == agent_player_id) & (df['Stage'] == 'preflop')].copy()
    if df_agent_preflop.empty: print("No preflop actions found for agent."); return pd.DataFrame(), {}

    # --- Ensure required columns exist ---
    log_cols = ['AgentPosition', 'IsRFIOpportunity', 'ObservationDict', 'Action', 'Hand']
    if not all(col in df_agent_preflop.columns for col in log_cols):
        missing = [col for col in log_cols if col not in df_agent_preflop.columns]
        print(f"Error: Missing required columns in log: {missing}. Analysis limited.")
        # Add defaults if possible, otherwise return
        if 'AgentPosition' not in df_agent_preflop.columns: df_agent_preflop['AgentPosition'] = 'Unknown'
        if 'IsRFIOpportunity' not in df_agent_preflop.columns: df_agent_preflop['IsRFIOpportunity'] = False # Assume not RFI if flag missing
        if 'Hand' not in df_agent_preflop.columns: df_agent_preflop['Hand'] = df_agent_preflop['ObservationDict'].apply(lambda x: x.get('hand', []))

    df_agent_preflop['Hand_Class'] = df_agent_preflop['Hand'].apply(classify_hand)
    df_agent_preflop = df_agent_preflop[df_agent_preflop['Hand_Class'] != "Unknown"]

    results = []
    analysis_summary = defaultdict(lambda: {'total': 0, 'matches': 0})
    skipped_situations = defaultdict(int)

    for _, row in df_agent_preflop.iterrows():
        hand_class = row['Hand_Class']
        player_pos = row['AgentPosition']
        agent_action_str = row['Action']

        # Determine situation (RFI or Facing RFI)
        # Use flag first, then attempt inference (which is currently limited)
        situation_type = None
        opponent_pos = None
        if row.get('IsRFIOpportunity', False):
            situation_type = 'RFI'
        else:
            # Attempt to infer if facing a raise (needs better logging)
            # For now, we'll mostly analyze explicit RFI spots
            # Pass 'Unknown' opponent if we guess it's Facing_RFI
            inferred_type, inferred_opp = determine_situation(row, agent_player_id)
            if inferred_type == 'Facing_RFI':
                 situation_type = 'Facing_RFI'
                 opponent_pos = 'Unknown' # Cannot determine from current logs
                 # print(f"Debug: Inferred Facing_RFI for {player_pos} with {hand_class}") # Optional debug

        if situation_type is None:
            skipped_situations['Other Preflop Spot'] += 1
            continue # Skip analysis for situations not covered

        # Get chart action
        chart_action_code = get_chart_action(situation_type, player_pos, opponent_pos, hand_class)

        if chart_action_code in ["Unknown Position", "Unknown Situation"]:
            skipped_situations[f"{situation_type}_{player_pos}_{opponent_pos}"] += 1
            continue # Skip if chart doesn't cover this specific spot

        # Map agent action
        agent_decision = 'F' # Default Fold
        if agent_action_str in ['bet_small', 'bet_big', 'all_in']:
            agent_decision = 'R' if situation_type == 'RFI' else '3B' # Raise or 3Bet
        elif agent_action_str == 'call':
            agent_decision = 'C' # Call
        elif agent_action_str == 'check':
            # Check is only valid RFI action if BB, otherwise treat as Call/Limp if SB
            if situation_type == 'RFI' and player_pos == 'SB':
                agent_decision = 'L' # Map Check/Call in SB RFI to Limp
            elif situation_type == 'Facing_RFI':
                 agent_decision = 'C' # Treat check facing raise as Call (though often illegal)
            else:
                 agent_decision = 'Check?' # Ambiguous

        # Determine match
        match = (agent_decision == chart_action_code)

        # --- Log Result ---
        analysis_key = f"{situation_type}_{player_pos}"
        if opponent_pos: analysis_key += f"_vs_{opponent_pos}"

        analysis_summary[analysis_key]['total'] += 1
        if match: analysis_summary[analysis_key]['matches'] += 1

        results.append({
            'Situation': analysis_key,
            'Hand_Class': hand_class,
            'Player_Pos': player_pos,
            'Opponent_Pos': opponent_pos or 'N/A',
            'Agent_Action': agent_action_str,
            'Agent_Decision_Mapped': agent_decision,
            'Chart_Action': chart_action_code,
            'Match': match
        })

    # --- Print Skipped/Summary ---
    if skipped_situations:
        print("\nWarning: Skipped analysis for some situations (not RFI or specific Facing RFI combo not encoded/determinable):")
        for sit, count in skipped_situations.items():
            print(f"- {sit}: {count} decisions")

    if not results:
        print("No decisions found matching encoded chart situations.")
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(results)

    print("\n--- Preflop Analysis Summary vs Charts ---")
    summary_list = []
    total_analyzed = 0
    total_matches = 0
    for sit, data in analysis_summary.items():
        rate = (data['matches'] / data['total'] * 100) if data['total'] > 0 else 0
        summary_list.append({
            'Situation': sit,
            'Decisions': data['total'],
            'Matches': data['matches'],
            'Adherence (%)': f"{rate:.2f}"
        })
        total_analyzed += data['total']
        total_matches += data['matches']

    overall_rate = (total_matches / total_analyzed * 100) if total_analyzed > 0 else 0
    print(f"Overall Analyzed Decisions: {total_analyzed}")
    print(f"Overall Matches: {total_matches}")
    print(f"Overall Adherence Rate: {overall_rate:.2f}%")
    summary_df = pd.DataFrame(summary_list)
    print(summary_df.to_string(index=False))


    # Analyze deviations
    deviations_df = results_df[~results_df['Match']]
    deviation_summary = deviations_df.groupby(['Situation', 'Hand_Class', 'Agent_Action', 'Chart_Action']).size().reset_index(name='Count')
    deviation_summary = deviation_summary.sort_values(by='Count', ascending=False)

    print("\n--- Top Deviations from Charts ---")
    print(deviation_summary.head(20))

    return summary_df, results_df, deviation_summary


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Decision Analysis for Poker RL Agent vs. Preflop Charts")
    parser.add_argument("--detailed_log", type=str, default="detailed_simulation_log.csv", help="Path to the detailed simulation log CSV file.")
    parser.add_argument("--output_analysis", type=str, default="decision_analysis_vs_charts.txt", help="Path to save the analysis summary report.")
    parser.add_argument("--agent_id", type=int, default=1, help="Player ID (1-based) of the agent to analyze.")
    args = parser.parse_args()

    print(f"Analyzing log file: {args.detailed_log}")
    try:
        df = pd.read_csv(args.detailed_log)
        print(f"Log loaded successfully. Rows: {len(df)}")
    except FileNotFoundError: print(f"Error: Log file not found at {args.detailed_log}"); return
    except Exception as e: print(f"Error reading log file: {e}"); return

    # --- Data Preprocessing ---
    required_cols = ['PlayerID', 'Action', 'ObservationDict', 'AgentPosition', 'IsRFIOpportunity']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in log file: {missing}")
        print("Ensure simulate.py logged 'AgentPosition' and 'IsRFIOpportunity'.")
        return

    df['ObservationDict'] = df['ObservationDict'].apply(parse_observation)
    df['Hand'] = df['ObservationDict'].apply(lambda x: x.get('hand', []))
    df['Stage'] = df['ObservationDict'].apply(lambda x: x.get('stage', '').lower())
    try:
        if df['IsRFIOpportunity'].dtype == object:
             df['IsRFIOpportunity'] = df['IsRFIOpportunity'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
        df['IsRFIOpportunity'] = df['IsRFIOpportunity'].astype(bool)
    except Exception as e:
        print(f"Warning: Could not convert 'IsRFIOpportunity' to boolean: {e}. Assuming False.")
        df['IsRFIOpportunity'] = False

    # --- Perform Analysis ---
    summary_df, details_df, deviations_df = analyze_preflop_vs_charts(df, agent_player_id=args.agent_id)

    # --- Save Output ---
    try:
        with open(args.output_analysis, "w") as f:
            f.write("--- Preflop Analysis Report vs Charts ---\n")
            f.write(f"Agent ID Analyzed: {args.agent_id}\n")
            f.write(f"Log File: {args.detailed_log}\n")
            f.write("Note: Compares agent actions to manually encoded charts (RFI subset + BTNvsUTG, BBvsBTN Facing RFI).\n")
            f.write("Note: Uses AgentPosition and IsRFIOpportunity flags from log.\n")
            f.write("Note: 'Facing RFI' analysis depends on accurate IsRFIOpportunity flag and may require more log context (opponent position) for full chart lookup.\n")

            f.write("\n--- Summary Statistics ---\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n--- Top Deviations from Charts ---")
            f.write(deviations_df.head(30).to_string(index=False))
            # Optionally write detailed adherence per hand:
            # f.write("\n\n--- Adherence Rate per Hand Class (Sample) ---\n")
            # f.write(details_df.groupby('Hand_Class')['Match'].mean().reset_index().sort_values(by='Match').head(20).to_string(index=False))

        print(f"\nAnalysis report saved to: {args.output_analysis}")
    except Exception as e:
        print(f"Error writing analysis report: {e}")

if __name__ == "__main__":
    main()
