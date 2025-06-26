"""Simple script to verify your Azure OpenAI credentials and deployment name.

Run `python -m util.test_azure_connection <DEPLOYMENT_NAME>`
or just `python util/test_azure_connection.py` if you exported the DEPLOYMENT_NAME
as AZURE_OPENAI_DEPLOYMENT.

It performs the smallest possible chat completion ("Hello") and dumps the raw
response so you can see whether authentication / routing is correct.
"""

import os
import sys
from pathlib import Path

from openai import AzureOpenAI, NotFoundError, AuthenticationError

# ---------------------------------------------------------------------------
# load .env if it exists so users don't have to export variables manually
# ---------------------------------------------------------------------------

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

DEPLOYMENT = (
    sys.argv[1] if len(sys.argv) > 1 else os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

if not DEPLOYMENT:
    print("❌  Please provide the deployment name as an arg or set "
          "AZURE_OPENAI_DEPLOYMENT in your .env file.")
    sys.exit(1)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

missing = [k for k, v in {"AZURE_OPENAI_API_KEY": api_key,
                          "AZURE_OPENAI_ENDPOINT": endpoint}.items() if v is None]
if missing:
    print("❌  Missing env vars:", ", ".join(missing))
    sys.exit(1)

print("🔧  Testing Azure OpenAI deployment …")
print("    endpoint   :", endpoint)
print("    deployment :", DEPLOYMENT)
print("    api version:", api_version)

client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
try:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            }
        ],
        max_tokens=2000,
    )
    print("✅  Success! Raw response:\n", resp)
except NotFoundError as e:
    print("❌  404 Not Found – Azure could not locate the deployment name you "
          "provided. Check the exact spelling in the portal, including case.")
    print(e)
except AuthenticationError as e:
    print("❌  Authentication failed – check AZURE_OPENAI_API_KEY and role." )
    print(e)
except Exception as e:
    print("❌  Unexpected error:")
    print(type(e).__name__, e)
