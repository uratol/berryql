Running the FastAPI demo

Prereqs
- Python 3.11+
- Install dependencies in a venv

Setup
1) Create and activate a virtual environment (if not already):
   - PowerShell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
2) Install requirements
   pip install -r requirements.txt

Config
- Edit .env to control DB and logging:
  - BERRYQL_TEST_DATABASE_URL: async SQLAlchemy URL; default is in-memory SQLite
  - SQL_ECHO=1: log SQL (set 0 to disable)
  - DEMO_SEED=1: seed demo data on startup

Run (VS Code)
- Press F5 and choose "FastAPI: Uvicorn (examples.main:app)"
- Or run the task: Terminal > Run Task > Run FastAPI (examples.main:app)

Run (PowerShell)
- With venv active:
  $env:SQL_ECHO=1; $env:DEMO_SEED=1; python -m uvicorn examples.main:app --reload --host 127.0.0.1 --port 8000

Open GraphiQL
- http://127.0.0.1:8000/graphql
