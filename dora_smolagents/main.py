import json
from smolagents import CodeAgent, Tool
from smolagents.models import LiteLLMModel

# --- Tools ---
class MoveForwardTool(Tool):
    name = "move_forward"
    description = "Move the robot forward"
    inputs = {}
    output_type = "object"

    def forward(self):
        return {"action": "move_forward"}

class StopTool(Tool):
    name = "stop_motion"
    description = "Stop the robot"
    inputs = {}
    output_type = "object"

    def forward(self):
        return {"action": "stop"}

# --- Agent setup ---
model = LiteLLMModel(
    model="ollama/mistral",
    api_base="http://localhost:11434"
)

agent = CodeAgent(
    tools=[MoveForwardTool(), StopTool()],
    model=model,
    max_steps=1
)

# --- Dora node ---
def handle_event(event):
    if event["type"] == "INPUT" and event["id"] == "command":
        command = event["value"]

        result = agent.run(command)

        # Extract clean action (fallback safe)
        if isinstance(result, dict):
            action = result
        else:
            action = {"action": "unknown"}

        return {
            "type": "OUTPUT",
            "id": "action",
            "value": json.dumps(action)
        }
