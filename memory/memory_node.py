from memory.state import AgentState
from typing import Dict, Any, Union
def memory_node(state: Union[AgentState, Dict[str, Any]]) -> AgentState:
    print(f"\n💾 Step 3: MEMORY - Archiving current decision...")
    if isinstance(state, dict):
        state = AgentState(
            action=state.get("action", ""),
            context=state.get("context", ""),
            retrieved_policies=state.get("retrieved_policies"),
            llm_decision=state.get("llm_decision"),
            memory=state.get("memory", []),
        )
        print("   Note: Input state was converted from dict to AgentState.")
    if state.memory is None:
        state.memory = []
    entry = {
        "action": state.action,
        "context": state.context,
        "result": state.llm_decision,
    }
    state.memory.append(entry)  
    print(f"   Action '{state.action[:30]}...' archived.")
    print(f"   Total decisions in memory: {len(state.memory)}")
    return state