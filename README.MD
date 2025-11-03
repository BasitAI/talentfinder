# 🚀 DATAMETRICS TalentFinder

**TalentFinder** is a Streamlit-based AI recruitment platform that searches, evaluates, and ranks candidates across 25+ job platforms using OpenAI and LangGraph.

---

## 🧰 Features
- 🔍 Multi-platform candidate search (LinkedIn, Indeed, Dice, Glassdoor, etc.)
- 🤖 AI-powered candidate evaluation (LLM + LangGraph)
- 📤 Multi-platform job posting
- 📊 Real-time analytics dashboard
- 📨 Automated email & LinkedIn invite messaging

---

## 🧑‍💻 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/talentfinder.git
cd talentfinder

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # (on macOS/Linux)
venv\Scripts\activate     # (on Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run locally
streamlit run app.py
