import json
import argparse
import os
from dora import Node
from smolagents import CodeAgent, Tool
from smolagents.models import LiteLLMModel

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

model = LiteLLMModel(
    model=os.getenv("MODEL", "ollama/mistral"),
    api_base=os.getenv("OLLAMA_BASE", "http://localhost:11434")
)

agent = CodeAgent(
    tools=[MoveForwardTool(), StopTool()],
    model=model,
    max_steps=1
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dora-smolagents")
    args = parser.parse_args()

    node = Node(args.name)

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "command":
            command = event["value"]
            result = agent.run(command)

            if isinstance(result, dict):
                action = result
            else:
                action = {"action": "unknown"}

            node.send_output(
                "action",
                json.dumps(action),
                event.get("metadata", {})
            )

if __name__ == "__main__":
    main()
