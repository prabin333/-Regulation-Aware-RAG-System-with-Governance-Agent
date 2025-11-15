from typing import TypedDict, List, Union, Dict, Any, Optional, Literal
from langchain_core.pydantic_v1 import BaseModel, Field
class GovernanceDecision(TypedDict):
    decision: Literal["Allowed", "Not Allowed", "Needs Review"]
    reason: str
    suggested_changes: List[str]
    references: List[str]
class AgentState(BaseModel):
    action: str = Field(description="The action the user proposes to take.")
    context: str = Field(description="The context and reason for the proposed action.")
    retrieved_policies: str = Field(default="", description="Relevant policy documents retrieved by the RAG system.")
    llm_decision: Optional[Dict[str, Any]] = Field(default=None, description="The structured GovernanceDecision output from the LLM.")
    memory: List[Dict[str, Any]] = Field(default_factory=list, description="Historical record of past actions and decisions.")