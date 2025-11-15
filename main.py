import os
import json
from graph import create_graph
from memory.state import AgentState
from typing import Dict, Any, Union 
def _read_input() -> AgentState:
    print("\nEnter input JSON (press ENTER twice to submit):")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return AgentState(action="")
        if line.strip() == "" and lines:
            break
        if line:
            lines.append(line)
    
    raw = "\n".join(lines).strip()
    if not raw:
        return AgentState(action="")
        
    try:
        obj = json.loads(raw)
        return AgentState(action=obj.get("action", ""), context=obj.get("context", ""))
    except Exception:

        return AgentState(action=raw, context="")


def main():
    policy_path = os.environ.get("POLICY_FILE", "data/data_policy.txt")
    model_path = os.environ.get("MODEL_PATH", "F:/ASSIGNMENT_1/model")

    graph = create_graph(policy_path=policy_path, model_path=model_path)
    print("🧠 Governance LangGraph Agent (with memory) ready.")

    while True:
        state = _read_input()
        if not state.action:
            print("👋 Exiting.")
            break

        try:
            result_raw = graph.invoke(state)
            result_state: AgentState
            if isinstance(result_raw, dict):
                result_state = AgentState(**result_raw)
            elif isinstance(result_raw, AgentState):
                result_state = result_raw
            else:
                raise TypeError(f"Unexpected graph output type: {type(result_raw)}")

            output: Union[Dict[str, Any], None] = result_state.llm_decision
            
            if output is None:
                output = {"error": "LLM decision was not generated or is missing."}

            print("\n🏁 Step 4: END - Workflow complete.")

            print("\n===== GOVERNANCE DECISION (Clean Output) =====")
            print(json.dumps(output, indent=2))
            print("=============================================\n")
            
        except Exception as e:
            print(f"❌ Error while processing: {e}")

if __name__ == "__main__":
    main()
