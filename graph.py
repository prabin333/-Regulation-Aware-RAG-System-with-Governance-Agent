from langgraph.graph import StateGraph, END
from memory.state import AgentState
from retriever import PolicyRetrieverNode
from governance_agent import EvaluateActionNode
from memory.memory_node import memory_node
def create_graph(policy_path: str, model_path: str):
    retriever_node = PolicyRetrieverNode(policy_path=policy_path)

    evaluate_node = EvaluateActionNode(model_path=model_path)

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retriever_node.run)
    workflow.add_node("evaluate", evaluate_node.run)
    workflow.add_node("memory", memory_node)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "evaluate")
    workflow.add_edge("evaluate", "memory")
    workflow.add_edge("memory", END)

    app = workflow.compile()

    print("✨ LangGraph compiled successfully.")
    return app
