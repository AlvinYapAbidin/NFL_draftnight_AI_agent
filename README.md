# Fantasy Football Draft Assistant

HERE WE GO, THE NFL SEASON IS BACK BABY 🏈🔥  
Your mission: **absolutely dominate** your fantasy football draft.  
Your weapon: an AI-powered assistant that thinks faster than your rival who reads mock drafts in the bathroom.  

Built with **LangGraph** + **Streamlit**, this bad boy works *without* Yahoo or ESPN API access — just plug in your picks manually, and watch AI craft your championship roster.

---

## Features That’ll Make Your League Tremble

- **Real-Time AI Draft Recommendations** – Custom-tailored picks for *your* roster needs, draft slot, and league dynamics.  
- **Value-Based Drafting (VBD)** – Sniffs out bargains like a veteran GM on clearance day.  
- **Position Scarcity Tracking** – Detect those position runs before they wipe the board clean.  
- **Adaptive Draft Strategy** – The AI evolves faster than a rookie in a contract year.  
- **No API Setup Needed** – Works with *any* fantasy platform.  
- **Live Draft Monitoring** – Always know what’s coming before it hits you.  
- **Team Export** – Download your final masterpiece as a CSV. Frame it. Brag about it.

---

## Installation (Preseason Warm-Up)

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Fantasy_Football_Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get your **OpenAI API key** — without it, your AI GM stays on the bench.

---

## Usage (Game Day Mode)

1. Run the app:
   ```bash
   streamlit run draft_assistant.py
   ```

2. Set up your league:
   - Enter your OpenAI API key in the sidebar  
   - Fill in league details (teams, draft slot, scoring format)  
   - Want custom player rankings? Drop your CSV into `data_players/`

3. Draft like a boss:
   - Log every pick in the **Draft Board** tab  
   - When it’s your turn, smash **Get Recommendations**  
   - Let AI serve you the winning move

4. Flex during and after:
   - **My Team tab**: See your roster rise to glory  
   - **Analysis tab**: Watch position trends like a Wall Street day trader

---

## Project Structure (Locker Room Tour)

```
Fantasy_Football_Agent/
├── draft_assistant.py       # Main Streamlit app
├── requirements.txt         # Playbook
├── pyproject.toml           # Team metadata
├── data_players/            # Player ranking CSVs
│   └── FantasyPros-expert-rankings.csv
└── README.md                # This hype-filled file
```

---

## Key Components

### DraftAgent (The Head Coach)
- Reads the field, calls the plays, makes you look like a genius.

### FantasyDataManager (The Scout)
- Brings in rankings, calculates VBD, spots value gems.

### Streamlit Interface (The Stadium)
- Keeps the fans (you) informed, engaged, and in control.
