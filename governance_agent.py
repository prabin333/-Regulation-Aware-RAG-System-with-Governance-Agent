from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import traceback
import re

class GovernanceAgent:
    def __init__(self, model_path="F:/ASSIGNMENT_1/model"):
        self.model_path = model_path
        self.pipe = None
        self.initialized = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model path not found: {self.model_path}")
                return
            
            print(f"🔹 Loading Gemma model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.initialized = True
            print("✅ Governance Agent initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            self.initialized = False
    
    def _create_prompt(self, action, context, retrieved_clauses):
        prompt = f"""<start_of_turn>user
You are a compliance governance agent. Analyze this action against the provided policy clauses.

ACTION: {action}
CONTEXT: {context}

POLICY CLAUSES:
{retrieved_clauses}

Provide your analysis in this EXACT JSON format (no additional text):

{{
    "decision": "Allowed|Not Allowed|Needs Review",
    "reason": "Brief clear reason referencing specific policy",
    "suggested_changes": [
        "Use a verified data processor.",
        "Encrypt data at rest and in transit.",
        "Obtain explicit consent for data processing."
    ],
    "references": [
        "Data Governance Policy Section X.X"
    ]
}}

IMPORTANT:
- Keep reason brief and clear (1-2 sentences)
- suggested_changes must be an array of specific actionable items
- references must be an array of specific policy sections
- Use the exact format above with no additional text<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _parse_json_response(self, text):
        try:
            cleaned_text = text.strip()
            
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = cleaned_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                if not isinstance(parsed.get("suggested_changes", []), list):
                    parsed["suggested_changes"] = [parsed["suggested_changes"]]
                if not isinstance(parsed.get("references", []), list):
                    parsed["references"] = [parsed["references"]]
                
                return parsed
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        
        return {
            "decision": "Needs Review",
            "reason": "Could not parse model response",
            "suggested_changes": ["Manual review required"],
            "references": ["N/A"]
        }
    
    def evaluate_action(self, action, context, retrieved_clauses):
        """Evaluate action compliance with policies"""
        if not self.initialized:
            return {
                "decision": "Error",
                "reason": "Model not initialized",
                "suggested_changes": ["Check model path and initialization"],
                "references": []
            }
        
        try:
            prompt = self._create_prompt(action, context, retrieved_clauses)
            
            result = self.pipe(
                prompt,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            output_text = result[0]["generated_text"].strip()
            print(f"🔹 Model raw output: {output_text[:200]}...")
            
            return self._parse_json_response(output_text)
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            traceback.print_exc()
            return {
                "decision": "Error",
                "reason": f"Generation failed: {str(e)}",
                "suggested_changes": ["Check model configuration"],
                "references": []
            }
