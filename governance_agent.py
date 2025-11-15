import os
from typing import List, Dict, Any, Literal, Union
from pydantic import BaseModel

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from memory.state import AgentState 

class GovernanceDecision(BaseModel):
    decision: Literal["Allowed", "Not Allowed", "Needs Review"]
    reason: str
    suggested_changes: List[str]
    references: List[str]

class EvaluateActionNode:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"⚠️ WARNING: Model folder not found at {model_path}. Using a **dummy LLM placeholder**.")
            class DummyLLM:
                def __call__(self, prompt):
                    return '{"decision": "Needs Review", "reason": "Model path not found, unable to evaluate compliance. Please set the correct model_path.", "suggested_changes": ["Configure the correct model_path in the notebook."], "references": ["N/A"]}'
            self.llm = DummyLLM()
        else:
            print("🤖 Loading local LLM (this may take a while)...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True, trust_remote_code=True
                )
                tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=None,
                    trust_remote_code=True,
                    local_files_only=True,
                )
        
                gen_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    return_full_text=False,
                )
        
                self.llm = HuggingFacePipeline(pipeline=gen_pipe)
                print("✅ EvaluateActionNode initialized.")
            except Exception as e:
                print(f"❌ Error loading model from {model_path}. Using a **dummy LLM placeholder**. Error: {e}")
                class DummyLLM:
                    def __call__(self, prompt):
                        return '{"decision": "Needs Review", "reason": "Model loading failed during initialization.", "suggested_changes": ["Check model path and dependencies."], "references": ["N/A"]}'
                self.llm = DummyLLM()
        self.parser = PydanticOutputParser(pydantic_object=GovernanceDecision)
        format_instructions = self.parser.get_format_instructions()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a Compliance Governance Agent.\n"
                    "Your ONLY task is to evaluate whether the ACTION complies with "
                    "the provided POLICIES.\n\n"
                    "RULES:\n"
                    "1) Only use the policies provided in 'RELEVANT POLICIES'.\n"
                    "2) Do NOT invent policies.\n"
                    "3) If unclear, return 'Needs Review'.\n"
                    "4) Output MUST follow the JSON schema.\n"
                    "5) Output ONLY valid JSON.\n"
                    "6) For 'references', cite the specific Policy Name AND Section Number (e.g., 'Data Governance Policy Section 4.2').\n"
                    "7) If the decision is 'Not Allowed' or 'Needs Review', the 'suggested_changes' list MUST contain actionable, compliant steps to mitigate the issues."
                )
            ),
            (
                "user",
                (
                    "ACTION:\n{action}\n\n"
                    "CONTEXT:\n{context}\n\n"
                    "RELEVANT POLICIES:\n{policies}\n\n"
                    "{format_instructions}"
                )
            ),
        ]).partial(format_instructions=format_instructions)


    def run(self, state: Union[AgentState, Dict[str, Any]]):
        print(f"\n🧠 Step 2: EVALUATE - Running LLM for decision...")

        def s(key, default=""):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        memory_block = ""
        memory_list = s("memory", [])

        if memory_list:
            lines = []
            for idx, item in enumerate(memory_list[-8:], start=1): 
                hist_a = item.get("action", "")
                hist_r = item.get("result", {})
                lines.append(f"Past {idx}: action='{hist_a}' | decision={hist_r.get('decision', 'N/A')}")
            memory_block = "\n\nPAST DECISIONS:\n" + "\n".join(lines)
            print(f"   Memory Used: {len(memory_list)} past decision(s) included in context.")

        prompt_input = {
            "action": s("action", ""),
            "context": s("context", "") + memory_block,
            "policies": s("retrieved_policies", ""),
        }
        
        print(f"   Policies: {s('retrieved_policies', '')[:50]}...")
        
        formatted_prompt = self.prompt.format(**prompt_input)
        llm_raw = self.llm(formatted_prompt)
        
        print(f"   LLM Output Received (raw): {llm_raw.strip()[:50]}...")

        try:
             parsed = GovernanceDecision(**json.loads(llm_raw.strip()))
        except Exception:
             parsed = self.parser.parse(llm_raw)

        parsed_dict = parsed.dict()
        if isinstance(state, dict):
            state["llm_decision"] = parsed_dict
        else:
            setattr(state, "llm_decision", parsed_dict)
            
        print(f"   Decision: '{parsed_dict.get('decision')}'")
        print("   Success: LLM decision parsed and added to state.")

        return state
