"""
Fantasy Football Draft Agent with LangGraph
No Yahoo API required - Manual input with AI recommendations
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Optional
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
import time


# ========================================
# DRAFT STATE MANAGEMENT
# ========================================

class DraftState(TypedDict):
    """State for the draft agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_pick: int
    round_num: int
    my_team: List[Dict[str, Any]]
    all_picks: List[Dict[str, Any]]
    available_players: pd.DataFrame
    position_needs: Dict[str, int]
    recommendations: List[Dict[str, Any]]
    draft_strategy: str
    league_settings: Dict[str, Any]
    next_pick_num: int


# ========================================
# FANTASY KNOWLEDGE BASE
# ========================================

DRAFT_STRATEGIES = [
    Document(page_content="""
    2025 Rookie-Focused Strategy:
    - Target proven 2024 rookies with upside
    - Jayden Daniels (QB) - Elite rushing floor, NFL ROY
    - Brock Bowers (TE) - Record-setting rookie, TE1
    - Malik Nabers (WR) - 20% target share, WR1 upside
    - Drake Maye (QB) - Undervalued rushing ability
    - Brian Thomas Jr. (WR) - Proven WR4 production
    """),
    Document(page_content="""
    Keeper League with Nico Collins Strategy:
    - Build around elite WR foundation (Collins keeper)
    - Target complementary WR2/WR3 (Nabers, BTJ)
    - Secure positional advantage at TE (Bowers)
    - Draft rushing QB for floor (Daniels/Maye)
    - Wait on RB - deeper rookie class emerging
    - Stack Collins with Texans QB if available
    """),
    Document(page_content="""
    2024 Rookie Value Strategy:
    - Marvin Harrison Jr. (WR25-30) - High draft capital, value
    - Rome Odunze (WR35-40) - Bears offense improving
    - Jonathon Brooks (RB25-30) - Clear path to touches
    - Travis Hunter - Undervalued vs Brian Thomas Jr.
    - Trey Benson - James Conner replacement upside
    """),
    Document(page_content="""
    Zero RB with 2025 Rookies:
    - Skip early RBs, load WR/TE
    - Target late-round rookie RBs with opportunity
    - Brooks, Benson have clear paths
    - 2024 rookie WRs more proven than RBs
    - Example: Collins(K)-Nabers-Bowers-BTJ-Brooks-Benson
    """),
    Document(page_content="""
    Dual-Threat QB Strategy 2025:
    - Rushing QBs dominated 2024 rookie class
    - Jayden Daniels - 891 rushing yards (record)
    - Caleb Williams - Dual-threat with weapons
    - Drake Maye - Undervalued rushing upside
    - J.J. McCarthy - Vikings weapons, rushing ability
    - Target early for positional advantage
    """),
    Document(page_content="""
    TE Premium Strategy 2025:
    - Brock Bowers broke rookie reception record
    - Massive gap after elite tier
    - Colston Loveland (Bears) - rookie upside
    - Target Bowers early for season-long advantage
    - TE scarcity more pronounced than ever
    """),
    Document(page_content="""
    Position Scarcity 2025 Update:
    - QB: Rookie class elite, rushing ability key
    - RB: Wait for opportunity-based rookies
    - WR: 2024 rookies proven, deep class
    - TE: Bowers tier 1, then massive cliff
    - DST: Stream based on matchups
    - Target positions with proven rookie success
    """),
]


# ========================================
# DATA LOADING AND PROCESSING
# ========================================

class FantasyDataManager:
    """Manages fantasy football data and rankings"""
    
    def __init__(self):
        self.rankings_df = None
        self.rankings_df = None
        self.projections_df = None
        
    def load_csv_from_folder(self) -> Optional[pd.DataFrame]:
        """Load CSV files from data_players folder"""
        import os
        import glob
        
        data_folder = "data_players"
        if not os.path.exists(data_folder):
            return None
            
        # Find CSV files in the folder
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if not csv_files:
            return None
            
        try:
            # Use the first CSV file found
            csv_file = csv_files[0]
            df = pd.read_csv(csv_file)
            
            # Standardize column names
            column_mapping = {
                'Player': 'Name',
                'player': 'Name',
                'Player Name': 'Name',
                'PLAYER': 'Name',
                'Pos': 'Position',
                'pos': 'Position',
                'POSITION': 'Position',
                'Team': 'Team',
                'team': 'Team',
                'TEAM': 'Team',
                'Rank': 'Rank',
                'rank': 'Rank',
                'ADP': 'Rank',
                'adp': 'Rank',
                'ECR': 'ECR',
                'Points': 'Projected_Points',
                'points': 'Projected_Points',
                'POINTS': 'Projected_Points',
                'Projected_Points': 'Projected_Points',
                'proj_points': 'Projected_Points',
                'Fantasy Points': 'Projected_Points',
                'fantasy_points': 'Projected_Points'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Ensure required columns exist
            required_cols = ['Name', 'Position', 'Team', 'Rank', 'Projected_Points']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Rank':
                        df[col] = range(1, len(df) + 1)
                    elif col == 'Projected_Points':
                        df[col] = 300 - (df.index * 1.5)
                    elif col == 'Team':
                        df[col] = 'Unknown'
                    elif col == 'Position':
                        df[col] = 'FLEX'
            
            # Add required columns for VBD calculation
            df['VBD'] = 0.0
            df['Tier'] = 0
            
            print(f"Successfully loaded {len(df)} players from {csv_file}")
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def load_default_rankings(self) -> pd.DataFrame:
        """Load rankings from CSV folder first, fallback to mock data"""
        # Try to load from CSV first
        csv_data = self.load_csv_from_folder()
        if csv_data is not None:
            return csv_data
            
        # Fallback to mock data
        print("No CSV found in data_players folder, using mock data")
        players = []
        
        # Top RBs half PPR(2025 Rankings)
        top_rbs = [
            ("Bijan Robinson", "RB", "ATL", 1, 350),      # Consensus RB1, led with 10 games of 20+ pts
            ("Saquon Barkley", "RB", "PHI", 2, 340),      # Led NFL in rushing yards (2,005), RB1 in all formats
            ("Jahmyr Gibbs", "RB", "DET", 3, 330),        # Elite efficiency, 7 games with 20+ fantasy pts
            ("Ashton Jeanty", "RB", "LV", 4, 320),        # Best RB prospect since Saquon, rookie phenom
            ("Christian McCaffrey", "RB", "SF", 5, 310),   # Injury concerns but still elite when healthy
            ("Derrick Henry", "RB", "BAL", 6, 300),        # RB4 last year, 16 rushing TDs
            ("De'Von Achane", "RB", "MIA", 7, 290),       # 22.6 PPG with Tua playing
            ("Jonathan Taylor", "RB", "IND", 8, 285),      # Still only 26, consistent RB1 upside
            ("Josh Jacobs", "RB", "GB", 9, 280),          # RB3 in Weeks 11-18, 15 TDs
            ("Bucky Irving", "RB", "TB", 10, 270),        # Breakout sophomore, receiving upside
            ("Kyren Williams", "RB", "LAR", 11, 265),     # Volume-based RB1 potential
            ("Kenneth Walker III", "RB", "SEA", 12, 260), # Still only 24 years old
            ("Chase Brown", "RB", "CIN", 13, 255),        # Ascending role, young talent
            ("James Cook", "RB", "BUF", 14, 250),         # Explosive in Buffalo offense
            ("Breece Hall", "RB", "NYJ", 15, 245),        # Bounce-back candidate with Rodgers
        ]
        
        # Top WRs half PPR(2025 Rankings including proven 2024 rookies)
        top_wrs = [
            ("Ja'Marr Chase", "WR", "CIN", 1, 360),       # 2024 Triple Crown winner, WR1 overall
            ("Justin Jefferson", "WR", "MIN", 2, 350),     # QB-proof, 92.4 YPG even with Darnold
            ("CeeDee Lamb", "WR", "DAL", 3, 340),         # Top-10 for 3 straight years, Pickens helps
            ("Malik Nabers", "WR", "NYG", 4, 330),        # Rookie record 109 catches, WR1 upside
            ("Brian Thomas Jr.", "WR", "JAC", 5, 320),    # 21.5 PPG final 7 games, elite finish
            ("Amon-Ra St. Brown", "WR", "DET", 6, 315),   # Safest floor in fantasy
            ("Puka Nacua", "WR", "LAR", 7, 310),          # 3.56 yards per route run, most efficient
            ("Nico Collins", "WR", "HOU", 8, 305),        # Top-3 in YPRR, Diggs gone
            ("A.J. Brown", "WR", "PHI", 9, 300),          # Elite when healthy, TD upside
            ("Drake London", "WR", "ATL", 10, 295),       # Breakout candidate with improved offense
            ("Ladd McConkey", "WR", "LAC", 11, 290),      # Herbert's top target, 100+ catch upside
            ("Tee Higgins", "WR", "CIN", 12, 285),        # Contract year, elite WR2
            ("Marvin Harrison Jr.", "WR", "ARI", 13, 280), # Year 2 breakout coming
            ("DK Metcalf", "WR", "PIT", 14, 275),         # Rodgers connection in Pittsburgh
            ("Terry McLaurin", "WR", "WAS", 15, 270),     # Career-best 13 TDs with Daniels
        ]
        
        # 2024 Rookie Stars
        rookie_stars = [
            ("Jayden Daniels", "QB", "WAS", 15, 280),  # NFL ROY
            ("Malik Nabers", "WR", "NYG", 18, 270),   # Target monster
            ("Brian Thomas Jr.", "WR", "JAX", 22, 260), # WR4 finish
            ("Brock Bowers", "TE", "LV", 25, 250),    # Record-breaking TE
            ("Caleb Williams", "QB", "CHI", 28, 240),  # 1st overall pick
            ("Drake Maye", "QB", "NE", 35, 220),      # Undervalued rushing
            ("Marvin Harrison Jr.", "WR", "ARI", 45, 200), # Value pick
            ("Rome Odunze", "WR", "CHI", 55, 180),    # Breakout candidate
            ("Jonathon Brooks", "RB", "CAR", 65, 160), # Clear path
            ("Trey Benson", "RB", "ARI", 75, 140),    # Conner replacement
        ]
        
        # Combine all top players including rookies
        all_players = top_rbs + top_wrs + rookie_stars
        
        # Extend with more players
        for i in range(11, 201):
            if i < 50:
                pos = ["RB", "WR"][i % 2]
            elif i < 80:
                pos = "WR"
            elif i < 100:
                pos = ["RB", "QB", "TE"][i % 3]
            else:
                pos = ["WR", "RB", "QB", "TE", "DST"][i % 5]
            
            player_name = f"Player {i}"
            team = ["KC", "BUF", "SF", "PHI", "CIN", "DAL", "MIA", "LAC"][i % 8]
            rank = i
            proj_points = 300 - (i * 1.5)
            
            all_players.append((player_name, pos, team, rank, proj_points))
        
        # Create DataFrame
        df = pd.DataFrame(all_players, columns=['Name', 'Position', 'Team', 'Rank', 'Projected_Points'])
        df['VBD'] = 0.0  # Will calculate later
        df['Tier'] = 0  # Will assign later
        
        return df
    
    def scrape_live_rankings(self) -> Optional[pd.DataFrame]:
        """Scrape current rankings from FantasyPros"""
        try:
            url = "https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find rankings table
            # This is simplified - actual scraping would be more complex
            rankings_data = []
            
            # Return None for now - would implement actual scraping
            return None
            
        except Exception as e:
            print(f"Error scraping rankings: {e}")
            return None
    
    def calculate_vbd(self, df: pd.DataFrame, league_size: int = 12) -> pd.DataFrame:
        """Calculate Value Based Drafting scores"""
        # Define replacement levels
        starters = {
            'QB': league_size * 1,
            'RB': league_size * 2.5,
            'WR': league_size * 3.5,
            'TE': league_size * 1,
            'DST': league_size * 1
        }
        
        # Calculate baselines
        for pos, num_starters in starters.items():
            pos_players = df[df['Position'] == pos].copy()
            if len(pos_players) > num_starters:
                baseline_idx = int(num_starters)
                baseline_points = pos_players.iloc[baseline_idx]['Projected_Points']
                df.loc[df['Position'] == pos, 'VBD'] = df.loc[df['Position'] == pos, 'Projected_Points'] - baseline_points
        
        return df
    
    def assign_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign players to tiers within positions"""
        for pos in df['Position'].unique():
            pos_players = df[df['Position'] == pos].copy()
            
            # Simple tier assignment based on projected points
            if len(pos_players) > 0:
                # Use clustering or manual breakpoints
                tier_breaks = np.percentile(pos_players['Projected_Points'], [80, 60, 40, 20])
                
                conditions = [
                    pos_players['Projected_Points'] >= tier_breaks[0],
                    pos_players['Projected_Points'] >= tier_breaks[1],
                    pos_players['Projected_Points'] >= tier_breaks[2],
                    pos_players['Projected_Points'] >= tier_breaks[3],
                ]
                
                choices = [1, 2, 3, 4]
                df.loc[df['Position'] == pos, 'Tier'] = np.select(conditions, choices, default=5)
        
        return df


# ========================================
# DRAFT AGENT WITH LANGGRAPH
# ========================================

class DraftAgent:
    """Main draft agent using LangGraph"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0.3
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.knowledge_base = self._setup_knowledge_base()
        self.data_manager = FantasyDataManager()
        
    def _setup_knowledge_base(self) -> FAISS:
        """Initialize vector store with draft strategies"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        splits = text_splitter.split_documents(DRAFT_STRATEGIES)
        return FAISS.from_documents(splits, self.embeddings)
    
    def analyze_draft_position(self, state: DraftState) -> DraftState:
        """Analyze current draft state and position needs"""
        my_team = state['my_team']
        
        # Count positions
        position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DST': 0}
        for player in my_team:
            if player['position'] in position_counts:
                position_counts[player['position']] += 1
        
        # Calculate needs based on league format
        league_settings = state['league_settings']
        roster_format = league_settings.get('roster_format', {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1, 'BENCH': 5
        })
        
        # Ideal roster construction (including FLEX considerations)
        ideal_counts = {
            'QB': 2,  # 1 starter + 1 backup
            'RB': 4,  # 2 starters + FLEX eligibility + bench
            'WR': 6,  # 3 starters + FLEX eligibility + bench
            'TE': 2,  # 1 starter + FLEX eligibility + backup
            'DST': 1
        }
        
        # Calculate needs
        position_needs = {}
        for pos, ideal in ideal_counts.items():
            current = position_counts.get(pos, 0)
            position_needs[pos] = max(0, ideal - current)
        
        state['position_needs'] = position_needs
        
        # Determine strategy based on picks
        round_num = state['round_num']
        if round_num <= 3:
            # Early rounds - focus on strategy
            if position_counts['RB'] == 0 and round_num > 2:
                state['draft_strategy'] = "Zero RB - Target WRs/TE"
            elif position_counts['RB'] >= 2:
                state['draft_strategy'] = "RB Heavy - Pivot to WR"
            else:
                state['draft_strategy'] = "Balanced BPA"
        else:
            # Mid-late rounds - fill needs
            biggest_need = max(position_needs.items(), key=lambda x: x[1])[0]
            state['draft_strategy'] = f"Fill needs - Target {biggest_need}"
        
        return state
    
    def calculate_pick_value(self, player: pd.Series, state: DraftState) -> float:
        """Calculate adjusted value for a player based on current state"""
        base_value = player['VBD']
        
        # Position need multiplier
        position = player['Position']
        need = state['position_needs'].get(position, 0)
        need_multiplier = 1.0 + (0.15 * min(need, 3))
        
        # Positional scarcity adjustment
        round_num = state['round_num']
        picks_until_next = state['next_pick_num'] - state['current_pick']
        
        scarcity_multiplier = 1.0
        if position == 'RB' and round_num <= 6:
            # RBs get scarce fast
            scarcity_multiplier = 1.1
        elif position == 'TE' and player['Tier'] <= 1:
            # Elite TE premium
            scarcity_multiplier = 1.15
        
        # Rank value - are we getting a discount?
        expected_pick = player['Rank']
        current_pick = state['current_pick']
        if current_pick > expected_pick + 5:
            # Great value
            value_multiplier = 1.2
        elif current_pick > expected_pick:
            # Good value
            value_multiplier = 1.1
        else:
            value_multiplier = 1.0
        
        total_value = base_value * need_multiplier * scarcity_multiplier * value_multiplier
        
        return total_value
    
    def generate_recommendations(self, state: DraftState) -> DraftState:
        """Generate draft recommendations"""
        available_players = state['available_players']
        
        # Calculate value for each available player
        values = []
        for idx, player in available_players.iterrows():
            value = self.calculate_pick_value(player, state)
            values.append({
                'player': player.to_dict(),
                'adjusted_value': value,
                'base_vbd': player['VBD'],
                'tier': player['Tier']
            })
        
        # Sort by value
        values.sort(key=lambda x: x['adjusted_value'], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for val in values[:15]:
            player = val['player']
            
            # Get strategy context
            strategy_context = self.knowledge_base.similarity_search(
                f"{state['draft_strategy']} {player['Position']} round {state['round_num']}",
                k=2
            )
            
            # Generate reasoning
            reasoning = self._generate_pick_reasoning(
                player, val, state, strategy_context
            )
            
            recommendations.append({
                'player': player,
                'adjusted_value': val['adjusted_value'],
                'base_vbd': val['base_vbd'],
                'tier': val['tier'],
                'reasoning': reasoning
            })
        
        state['recommendations'] = recommendations[:10]
        return state
    
    def _generate_pick_reasoning(self, player: Dict, value_data: Dict, 
                                state: DraftState, context: List[Document]) -> str:
        """Generate detailed reasoning for a pick"""
        context_text = "\n".join([doc.page_content for doc in context])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert fantasy football analyst providing draft advice."),
            ("human", """
            Context: {context}
            
            Current Draft State:
            - Pick: #{current_pick} (Round {round_num})
            - My Team: {my_team}
            - Position Needs: {position_needs}
            - Strategy: {draft_strategy}
            - Picks until next: {picks_until_next}
            
            Player: {player_name} ({position}, {team})
            - Rank: {rank}
            - Projected Points: {proj_points}
            - VBD Score: {vbd:.1f}
            - Tier: {tier}
            - Adjusted Value: {adj_value:.1f}
            
            In 2-3 sentences, explain why this player would be a good pick here.
            Focus on value, team need, and positional scarcity.
            """)
        ])
        
        chain = prompt | self.llm
        
        my_team_summary = [f"{p['name']} ({p['position']})" for p in state['my_team']]
        
        response = chain.invoke({
            'context': context_text,
            'current_pick': state['current_pick'],
            'round_num': state['round_num'],
            'my_team': ", ".join(my_team_summary) if my_team_summary else "Empty",
            'position_needs': state['position_needs'],
            'draft_strategy': state['draft_strategy'],
            'picks_until_next': state['next_pick_num'] - state['current_pick'],
            'player_name': player['Name'],
            'position': player['Position'],
            'team': player['Team'],
            'rank': player['Rank'],
            'proj_points': player['Projected_Points'],
            'vbd': value_data['base_vbd'],
            'tier': value_data['tier'],
            'adj_value': value_data['adjusted_value']
        })
        
        return response.content
    
    def create_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(DraftState)
        
        # Add nodes
        workflow.add_node("analyze_position", self.analyze_draft_position)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        
        # Add edges
        workflow.add_edge(START, "analyze_position")
        workflow.add_edge("analyze_position", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        return workflow.compile()


# ========================================
# STREAMLIT UI
# ========================================

def init_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.all_picks = []
        st.session_state.my_team = []
        st.session_state.current_pick = 1
        st.session_state.data_manager = FantasyDataManager()
        st.session_state.player_rankings = st.session_state.data_manager.load_default_rankings()
        st.session_state.draft_complete = False
        st.session_state.agent = None
        st.session_state.graph = None


def calculate_next_pick(current_pick: int, my_position: int, league_size: int) -> int:
    """Calculate when your next pick will be (snake draft)"""
    current_round = ((current_pick - 1) // league_size) + 1
    position_in_round = ((current_pick - 1) % league_size) + 1
    
    if current_round % 2 == 1:  # Odd round (forward)
        if position_in_round < my_position:
            # Haven't picked yet this round
            next_pick = (current_round - 1) * league_size + my_position
        else:
            # Already picked this round or past my position, go to next round
            next_round = current_round + 1
            if next_round % 2 == 1:  # Next round is odd (forward)
                next_pick = (next_round - 1) * league_size + my_position
            else:  # Next round is even (reverse)
                next_pick = next_round * league_size - my_position + 1
    else:  # Even round (reverse)
        reverse_position = league_size - my_position + 1
        if position_in_round < reverse_position:
            # Haven't picked yet this round
            next_pick = current_round * league_size - my_position + 1
        else:
            # Already picked this round, go to next round
            next_round = current_round + 1
            if next_round % 2 == 1:  # Next round is odd (forward)
                next_pick = (next_round - 1) * league_size + my_position
            else:  # Next round is even (reverse)
                next_pick = next_round * league_size - my_position + 1
    
    return next_pick


def main():
    st.set_page_config(
        page_title="Fantasy Draft Assistant",
        page_icon="🏈",
        layout="wide"
    )
    
    init_session_state()
    
    # Header
    st.title("🏈 Fantasy Football Draft Assistant")
    st.markdown("*Powered by LangGraph - No Yahoo API Required*")
    
    # Sidebar - Setup
    with st.sidebar:
        st.header("⚙️ Draft Settings")
        
        # API Key
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
        
        if api_key and not st.session_state.agent:
            try:
                st.session_state.agent = DraftAgent(api_key)
                st.session_state.graph = st.session_state.agent.create_graph()
                st.success("✅ AI Agent initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing AI agent: {str(e)}")
                st.session_state.agent = None
                st.session_state.graph = None
        
        # League Settings
        st.subheader("League Info")
        league_size = st.number_input("Teams", 8, 16, 12)
        my_position = st.number_input("Your Draft Position", 1, league_size, 1)
        scoring = st.selectbox("Scoring", ["Half-PPR", "PPR", "Standard"])
        
        # Save settings
        if 'league_settings' not in st.session_state:
            st.session_state.league_settings = {
                'league_size': league_size,
                'my_position': my_position,
                'scoring': scoring,
                'roster_format': {
                    'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 
                    'FLEX': 1, 'DST': 1, 'BENCH': 5
                }
            }
        
        # Quick Actions
        st.subheader("🎯 Quick Actions")
        
        if st.button("🔄 Reset Draft", type="secondary"):
            st.session_state.all_picks = []
            st.session_state.my_team = []
            st.session_state.current_pick = 1
            st.session_state.draft_complete = False
            st.rerun()
    
    # Main Content
    if not api_key:
        st.warning("👈 Please enter your OpenAI API key in the sidebar to get started")
        st.stop()
    
    # Current Pick Info
    col1, col2, col3, col4 = st.columns(4)
    current_pick = st.session_state.current_pick
    round_num = ((current_pick - 1) // league_size) + 1
    pick_in_round = ((current_pick - 1) % league_size) + 1
    
    col1.metric("Current Pick", f"#{current_pick}")
    col2.metric("Round", f"{round_num}")
    col3.metric("Pick in Round", f"{pick_in_round}")
    
    # Calculate next pick
    next_pick = calculate_next_pick(current_pick, my_position, league_size)
    picks_until_next = next_pick - current_pick
    col4.metric("Your Next Pick", f"#{next_pick} ({picks_until_next} picks)")
    
    # Check if it's user's turn
    is_my_turn = False
    if round_num % 2 == 1:  # Odd round
        is_my_turn = pick_in_round == my_position
    else:  # Even round
        is_my_turn = pick_in_round == (league_size - my_position + 1)
    
    if is_my_turn:
        st.success("🎯 **IT'S YOUR TURN TO PICK!**")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Draft Board", "🤖 Recommendations", "📊 My Team", "📈 Analysis"])
    
    with tab1:
        st.subheader("Draft Board")
        
        # Pick entry
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        player_name = col1.text_input("Player Name", key="pick_entry")
        position = col2.selectbox("Position", ["RB", "WR", "QB", "TE", "DST"], key="pick_position")
        team = col3.text_input("Team", key="pick_team")
        
        if col4.button("➕ Add Pick", type="primary", disabled=not player_name):
            # Add pick
            pick_data = {
                'pick_number': current_pick,
                'round': round_num,
                'player': player_name,
                'position': position,
                'team': team,
                'is_mine': is_my_turn,
                'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.all_picks.append(pick_data)
            
            if is_my_turn:
                pick_data['name'] = player_name
                st.session_state.my_team.append(pick_data)
            
            st.session_state.current_pick += 1
            
            # Clear inputs
            st.rerun()
        
        # Show recent picks
        if st.session_state.all_picks:
            recent_picks = st.session_state.all_picks[-10:][::-1]
            
            st.markdown("### Recent Picks")
            for pick in recent_picks:
                if pick['is_mine']:
                    st.markdown(f"**{pick['pick_number']}. {pick['player']} ({pick['position']}, {pick['team']}) ⭐**")
                else:
                    st.markdown(f"{pick['pick_number']}. {pick['player']} ({pick['position']}, {pick['team']})")
    
    with tab2:
        st.subheader("🤖 AI Recommendations")
        
        if st.button("🔮 Get Recommendations", type="primary"):
            if not st.session_state.agent or not st.session_state.graph:
                st.error("❌ AI Agent not initialized. Please check your API key.")
            else:
                # Get drafted players
                drafted_names = [p['player'] for p in st.session_state.all_picks]
                
                # Update available players
                all_players = st.session_state.data_manager.load_default_rankings()
                all_players = st.session_state.data_manager.calculate_vbd(all_players, league_size)
                all_players = st.session_state.data_manager.assign_tiers(all_players)
                available = all_players[~all_players['Name'].isin(drafted_names)]
                
                # Create state
                state = DraftState(
                    messages=[],
                    current_pick=current_pick,
                    round_num=round_num,
                    my_team=st.session_state.my_team,
                    all_picks=st.session_state.all_picks,
                    available_players=available,
                    position_needs={},
                    recommendations=[],
                    draft_strategy="",
                    league_settings=st.session_state.league_settings,
                    next_pick_num=next_pick
                )
                
                # Run agent
                with st.spinner("Analyzing draft and generating recommendations..."):
                    result = st.session_state.graph.invoke(state)
                
                # Display results
                st.info(f"**Strategy:** {result['draft_strategy']}")
                
                # Position needs
                needs = result['position_needs']
                need_badges = []
                for pos, count in needs.items():
                    if count > 0:
                        need_badges.append(f"{pos}: {count}")
                
                if need_badges:
                    st.warning(f"**Position Needs:** {', '.join(need_badges)}")
                
                # Recommendations
                st.markdown("### Top Recommendations")
                
                for i, rec in enumerate(result['recommendations'][:5]):
                    player = rec['player']
                    
                    with st.expander(
                        f"#{i+1} - **{player['Name']}** ({player['Position']}, {player['Team']}) "
                        f"- Tier {int(rec['tier'])}"
                    ):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rank", f"#{int(player['Rank'])}")
                        col2.metric("Proj Points", f"{player['Projected_Points']:.0f}")
                        col3.metric("VBD", f"{rec['base_vbd']:.1f}")
                        col4.metric("Value Score", f"{rec['adjusted_value']:.1f}")
                        
                        st.markdown("**Analysis:**")
                        st.write(rec['reasoning'])
                        
                        # Quick add button
                        if st.button(f"Draft {player['Name']}", key=f"draft_{i}"):
                            # Add this player
                            pick_data = {
                                'pick_number': current_pick,
                                'round': round_num,
                                'player': player['Name'],
                                'position': player['Position'],
                                'team': player['Team'],
                                'is_mine': True,
                                'name': player['Name'],
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            st.session_state.all_picks.append(pick_data)
                            st.session_state.my_team.append(pick_data)
                            st.session_state.current_pick += 1
                            st.rerun()
    
    with tab3:
        st.subheader("📊 My Team")
        
        if st.session_state.my_team:
            # Position breakdown
            pos_counts = {}
            for player in st.session_state.my_team:
                pos = player['position']
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            # Display position counts
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("QB", pos_counts.get('QB', 0))
            col2.metric("RB", pos_counts.get('RB', 0))
            col3.metric("WR", pos_counts.get('WR', 0))
            col4.metric("TE", pos_counts.get('TE', 0))
            col5.metric("DST", pos_counts.get('DST', 0))
            
            # Calculate FLEX eligible players (RB + WR + TE)
            flex_eligible = pos_counts.get('RB', 0) + pos_counts.get('WR', 0) + pos_counts.get('TE', 0)
            col6.metric("FLEX Eligible", flex_eligible)
            
            # Team roster
            st.markdown("### Roster")
            team_df = pd.DataFrame(st.session_state.my_team)
            display_df = team_df[['pick_number', 'round', 'player', 'position', 'team']]
            display_df.columns = ['Pick #', 'Round', 'Player', 'Pos', 'Team']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export option
            if st.button("📥 Export Team"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fantasy_team_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No players drafted yet. Start drafting to build your team!")
    
    with tab4:
        st.subheader("📈 Draft Analysis")
        
        if st.session_state.all_picks:
            # Draft trends
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Position Run Analysis")
                
                # Calculate position runs
                position_runs = []
                for i in range(max(0, len(st.session_state.all_picks) - 10), len(st.session_state.all_picks)):
                    if i >= 0:
                        pick = st.session_state.all_picks[i]
                        position_runs.append(pick['position'])
                
                if position_runs:
                    run_counts = pd.Series(position_runs).value_counts()
                    st.bar_chart(run_counts)
                    
                    # Check for runs
                    for pos, count in run_counts.items():
                        if count >= 3:
                            st.warning(f"⚠️ {pos} run detected! {count} {pos}s in last 10 picks")
            
            with col2:
                st.markdown("### Draft Pace")
                
                # Show picks by position over time
                picks_by_round = {}
                for pick in st.session_state.all_picks:
                    round_num = pick['round']
                    if round_num not in picks_by_round:
                        picks_by_round[round_num] = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}
                    
                    pos = pick['position']
                    if pos in picks_by_round[round_num]:
                        picks_by_round[round_num][pos] += 1
                
                # Create visualization
                if picks_by_round:
                    rounds_df = pd.DataFrame.from_dict(picks_by_round, orient='index')
                    rounds_df.index.name = 'Round'
                    st.line_chart(rounds_df[['RB', 'WR', 'QB', 'TE']])
            
            # Value analysis
            st.markdown("### Value Picks & Reaches")
            
            # This would show Rank vs actual draft position
            # For now, showing placeholder
            st.info("Track which players went above/below their rank as the draft progresses")
        
        else:
            st.info("No picks made yet. Start drafting to see analysis!")
    
    # Footer with tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        **Keyboard Shortcuts (coming soon):**
        - `Enter` - Add current player
        - `Tab` - Next field
        - `R` - Get recommendations
        
        **Draft Strategy Tips:**
        - Don't panic if you miss a player - there's always value later
        - Position runs create opportunity at other positions
        - Track your next pick to plan ahead
        - Use tiers, not rankings - players in same tier are similar value
        
        **Using Recommendations:**
        - The AI considers your roster construction, position scarcity, and value
        - Higher "Value Score" = better pick for your team's needs
        - Don't always take #1 recommendation - consider your strategy
        """)


# ========================================
# QUICK DRAFT SIMULATOR
# ========================================

class DraftSimulator:
    """Simulate other teams' picks for testing"""
    
    def __init__(self, rankings_df: pd.DataFrame):
        self.rankings = rankings_df
        self.drafted = set()
    
    def simulate_pick(self, pick_num: int, team_needs: Dict[str, int]) -> Dict:
        """Simulate a CPU pick"""
        available = self.rankings[~self.rankings['Name'].isin(self.drafted)]
        
        # Simple logic - take best available with small need adjustment
        scores = []
        for _, player in available.head(20).iterrows():
            score = player['Rank'] * -1  # Lower Rank = better
            
            # Small position need bonus
            pos = player['Position']
            if team_needs.get(pos, 0) > 0:
                score += 5
            
            scores.append((score, player))
        
        # Pick best score
        scores.sort(key=lambda x: x[0], reverse=True)
        picked_player = scores[0][1]
        
        self.drafted.add(picked_player['Name'])
        
        return {
            'player': picked_player['Name'],
            'position': picked_player['Position'],
            'team': picked_player['Team']
        }


# ========================================
# DATA IMPORT HELPERS
# ========================================

def import_rankings_csv(file_path: str) -> pd.DataFrame:
    """Import rankings from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'Player': 'Name',
            'Pos': 'Position',
            'Team': 'Team',
            'Rank': 'Rank',
            'Points': 'Projected_Points'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns
        required_cols = ['Name', 'Position', 'Team', 'Rank', 'Projected_Points']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Projected_Points':
                    # Estimate if missing
                    df[col] = 300 - (df['Rank'] * 1.5)
                else:
                    df[col] = 'Unknown'
        
        return df
        
    except Exception as e:
        st.error(f"Error importing CSV: {e}")
        return None


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # Run Streamlit app
    main()


# ========================================
# USAGE INSTRUCTIONS
# ========================================

"""
HOW TO USE THIS DRAFT ASSISTANT:

1. Setup (5 minutes):
   - Install requirements: pip install -r requirements.txt
   - Get OpenAI API key from https://platform.openai.com
   - Run: streamlit run draft_assistant.py

2. Pre-Draft (10 minutes):
   - Enter your league settings (size, position, scoring)
   - Optional: Import custom rankings CSV
   - Review the AI's knowledge base

3. During Draft:
   - Enter each pick as it happens (takes 5 seconds)
   - When it's your turn, click "Get Recommendations"
   - Review top 5 suggestions with AI reasoning
   - Either click player name to draft or enter manually
   - Track position runs and draft trends

4. Key Features:
   - No Yahoo login required
   - Real-time position need analysis
   - Value-based drafting (VBD) calculations
   - Tier-based recommendations
   - Position scarcity awareness
   - Export your team anytime

5. Tips:
   - The AI adjusts strategy based on your picks
   - Watch for position runs (3+ same position)
   - Track when your next pick is
   - Don't reach for needs - take value

ADVANTAGES OVER API APPROACH:
- Works immediately (no OAuth setup)
- Never breaks during draft
- Full control over player pool
- Can add keepers/custom values
- Works with any platform (ESPN, Sleeper, etc.)
"""

# streamlit run draft_assistant.py
