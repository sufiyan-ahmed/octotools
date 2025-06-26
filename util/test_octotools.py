import os
import sys
import traceback
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
print("✅ .env file loaded successfully")

# Print environment variables (masking sensitive data)
print("\n=== Environment Variables ===")
print(f"AZURE_OPENAI_API_KEY set: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
print(f"AZURE_OPENAI_ENDPOINT set: {'Yes' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'No'}")
print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION', 'Not set')}")

# Import the solver
try:
    print("\n=== Importing Solver ===")
    from octotools.solver import construct_solver
    print("✅ Solver imported successfully")
except Exception as e:
    print(f"❌ Error importing solver: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Construct the solver
try:
    print("\n=== Constructing Solver ===")
    deployment_name = "azure-"+ os.getenv("AZURE_OPENAI_DEPLOYMENT")
    print(f"Creating solver with model: {deployment_name}")
    solver = construct_solver(llm_engine_name=deployment_name)
    print("✅ Solver constructed successfully")
except Exception as e:
    print(f"❌ Error constructing solver: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Test the solver
try:
    print("\n=== Testing Solver ===")
    print("Sending test query: 'What is the capital of France?'")
    
    # First try a simple query
    output = solver.solve("What is the capital of France?")
    
    print("\n=== Response ===")
    print(f"Response type: {type(output)}")
    if isinstance(output, dict):
        for key, value in output.items():
            print(f"{key}: {value}")
    else:
        print(f"Unexpected response format: {output}")
    
except Exception as e:
    print(f"\n❌ Error during solver execution: {str(e)}")
    traceback.print_exc()