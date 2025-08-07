# Fantasy Football Draft Assistant
An AI-powered assistant that delivers real-time draft recommendations and analysis to help you dominate your fantasy football draft. Built with LangGraph and Streamlit, this tool works seamlessly without requiring Yahoo or ESPN API access — simply input picks manually and let AI guide your strategy.


## Features
Real-Time AI Draft Recommendations - Personalized suggestions based on your roster needs, draft position, and league trends.
Value-Based Drafting (VBD) - Calculates the best value picks based on positional scarcity and player rankings.
Position Scarcity Tracking - Detect when position runs are happening and adjust your strategy accordingly.
Adaptive Draft Strategy - AI evolves your strategy based on picks made and team composition.
No API Setup Needed - Works with any fantasy platform — just enter draft picks manually.
Live Draft Monitoring - Track your next pick, position trends, and overall draft dynamics.
Team Export - Download your final roster as a CSV for easy reference or sharing.


## Installation
1. Clone this repository:

	git clone <repository-url>
	cd Fantasy_Football_Agent

2. Install dependencies:

	pip install -r requirements.txt

3. Get your OpenAI API key

## Usage
1. Run the app:

	streamlit run draft_assistant.py

2. Configure your draft:
• Enter your OpenAI API key in the sidebar
• 	Set league details (e.g. number of teams, draft slot, scoring format)
• 	Optionally place a custom player rankings CSV into the data_players/ folder
3.During the draft:
• 	Enter each pick in the Draft Board tab
• 	When it’s your turn, head to Recommendations and click “Get Recommendations”
• 	Review AI insights and select your next player
4. Track your progress:
• 	Monitor your team under the My Team tab
• 	Review draft trends and positional runs in the Analysis tab


## Project Structure

	Fantasy_Football_Agent/
	├── draft_assistant.py       # Main Streamlit app
	├── requirements.txt         # Dependency list
	├── pyproject.toml           # Project metadata
	├── data_players/            # Folder for player ranking CSVs
	│   └── FantasyPros-expert-rankings.csv
	└── README.md                # This file


## Key Components

DraftAgent
The core LangGraph-powered agent that:
• 	Analyzes live draft status and team needs
• 	Provides high-value player recommendations
• 	Dynamically adapts strategy

## FantasyDataManager

Manages fantasy player data:
• 	Loads and parses ranking CSVs
• 	Computes VBD and tiered rankings
• 	Ensures flexibility between mock and real data

## Streamlit Interface

User-friendly frontend for:
• 	Pick input and draft board view
• 	AI-driven suggestions and reasoning
• 	Team and trend analysis
 
Draft Strategies Included
•	Rookie Upside – Target 2024 rookies with breakout potential
•	Keeper League Focus – Build around key keepers like Nico Collins
•	Zero RB – Prioritize WRs/TEs early, draft RBs for value
•	Dual-Threat QB – Secure rushing QBs for weekly advantage
•	TE Premium – Target elite TEs early in bonus formats

## Data Sources

Supports:
• 	Custom player rankings via CSV (/data_players/)
• 	Preloaded 2025 rankings (mock)
• 	Real-time projections and position trends via AI

## Requirements
• 	Python 3.11+
• 	OpenAI API key
• 	Streamlit
• 	LangChain / LangGraph
• 	Pandas, NumPy
• 	FAISS

## Contributing
1.	Fork the repository
2.	Create a feature branch
3.	Commit and push changes
4.	Add tests (if applicable)
5.	Submit a pull request

## License

MIT License — use freely and modify as needed.

## Troubleshooting
•	Is your OpenAI API key valid?
•	Are all required packages installed?
•	Are you using Python 3.11+?

If you encounter issues, please open an issue on GitHub with the error details.

## Note: This tool is for entertainment and educational use. All draft decisions are ultimately your responsibility.

Let me know if you’d like this exported as a .md file or further tailored.
