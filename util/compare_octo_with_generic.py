"""
Compare a direct Azure OpenAI chat completion with the same query answered
via Octotools solver.  Both responses are stored under ./outputs/<run-id>_*.txt.

Usage:
    python util/compare_octo_with_generic.py "your prompt here" or modify the DEFAULT_QUERY variable
Set VERBOSE=true to see progress logs.
"""
import os, sys, uuid, json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
print("\n=== Loading Environment Variables ===")
print(f"Current working directory: {os.getcwd()}")
dotenv_path = os.path.join(project_root, '.env')
print(f"Looking for .env file at: {dotenv_path}")

if not os.path.exists(dotenv_path):
    print("❌ Error: .env file not found!")
    sys.exit(1)

# Load the .env file
import dotenv
dotenv.load_dotenv(dotenv_path, override=True)
print("✅ .env file loaded successfully")                   # expect AZURE_* vars


DEFAULT_QUERY = "Choose a number between 1 to 50"
QUERY = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
DIR_NAME = "TestFolder"
OUTPUT_DIR = Path(__file__).parent / "outputs" / DIR_NAME
OUTPUT_DIR.mkdir(exist_ok=True)


def log(msg: str):
    if os.getenv("VERBOSE","false").lower() in ("1","true","yes"):
        print(msg)

# ---- Generic Azure OpenAI call ------------------------------------------------
import openai
client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01-preview"),
)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
log(f"Calling Azure Chat completion (deployment={deployment}) …")
generic_resp = client.chat.completions.create(
    model=deployment,
    messages=[{"role":"user","content":QUERY}],
)
generic_answer = generic_resp.choices[0].message.content.strip()
(OUTPUT_DIR/f"{RUN_ID}_generic.txt").write_text(generic_answer, encoding="utf-8")

# ---- Octotools solver call ----------------------------------------------------
log("Calling Octotools solver …")
from octotools.solver import construct_solver
solver = construct_solver(llm_engine_name='azure-' + deployment)
octo_out = solver.solve(QUERY)

# If solver returns dict, pick “answer” key; else str
octo_answer = octo_out.get("direct_output") if isinstance(octo_out, dict) else str(octo_out)
(OUTPUT_DIR/f"{RUN_ID}_octo.txt").write_text(octo_answer.strip(), encoding="utf-8")

log(f"✅ Finished.  Files written with run-id {RUN_ID} in {OUTPUT_DIR}")