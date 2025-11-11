import json
import sys
import re
from governance_agent import GovernanceAgent
from retriever import PolicyRetriever

class GovernanceRAGSystem:
    def __init__(self):
        print("🧠 Initializing Governance RAG System...")

        self.retriever = PolicyRetriever("data/data_policy.txt")
        self.agent = GovernanceAgent()
        
        print("✅ System ready!\n")
    
    def parse_user_input(self):
        print("=" * 60)
        print("Governance RAG System - Input")
        print("=" * 60)
        print("Enter your query in one of these formats:")
        print("1. JSON: {{\"action\": \"...\", \"context\": \"...\"}}")
        print("2. Simple: Action description [Context description]")
        print("3. Press Enter twice to finish\n")
        
        lines = []
        while True:
            try:
                line = input().strip()
                if line == "" and lines:
                    break
                if line:
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        
        user_input = "\n".join(lines)

        try:
            data = json.loads(user_input)
            action = data.get("action", "").strip()
            context = data.get("context", "").strip()
        except json.JSONDecodeError:
            if "[" in user_input and "]" in user_input:
                action_part = user_input.split("[")[0].strip()
                context_match = re.search(r"\[(.*?)\]", user_input)
                context = context_match.group(1) if context_match else ""
                action = action_part
            else:
                action = user_input
                context = ""
        
        return action, context
    
    def process_query(self, action, context):
        """Process a single query through the RAG pipeline"""
        if not action:
            print("❌ No action provided")
            return None
        
        print(f"\n🔍 Analyzing: {action}")
        if context:
            print(f"📋 Context: {context}")
        
        print("\n📚 Retrieving relevant policy clauses...")
        retrieved_clauses = self.retriever.retrieve(action, k=4)
        
        if not retrieved_clauses:
            print("❌ No relevant policies found")
            return None
        
        print("\n📖 Retrieved Policy Clauses:")
        for i, clause in enumerate(retrieved_clauses, 1):
            print(f"  {i}. [{clause['reference']}] Score: {clause['score']:.3f}")
            print(f"     {clause['text'][:100]}...")
        
        clauses_text = "\n".join([
            f"- {clause['reference']}: {clause['text']}" 
            for clause in retrieved_clauses
        ])
        
        print("\n⚖️ Consulting governance agent...")
        decision = self.agent.evaluate_action(action, context, clauses_text)
        
        return decision
    
    def run_interactive(self):
        """Run the system in interactive mode"""
        while True:
            try:
                action, context = self.parse_user_input()
                
                if not action:
                    print("👋 Exiting...")
                    break
                
                decision = self.process_query(action, context)
                
                if decision:
                    print("\n" + "=" * 60)
                    print("✅ FINAL GOVERNANCE DECISION (JSON):")
                    print("=" * 60)
                    print(json.dumps(decision, indent=2))
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    """Main entry point"""
    try:
        system = GovernanceRAGSystem()
        system.run_interactive()
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
