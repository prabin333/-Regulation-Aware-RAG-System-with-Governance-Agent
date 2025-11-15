# 📘 Regulation-Aware RAG System with Governance Agent (with Memory)

This project implements a **Regulation-Aware Retrieval-Augmented Generation (RAG) System** combined with a **Compliance Governance Agent** using Langchain and LangGraph.
It evaluates whether a proposed **ACTION** complies with an organization’s **policies**, explains the reasoning, and suggests changes if necessary.

The system also includes an **AI Memory Module**, enabling context-aware decisions based on past actions.

# ✨ Key Features

### 🔍 **1. Policy Retrieval (RAG)**

* Loads policy documents
* Splits into chunks
* Embeds using `sentence-transformers/all-MiniLM-L6-v2`
* Builds or loads FAISS index
* Retrieves top-k relevant policy sections

### 🧠 **2. Governance Evaluation**

* Evaluates ACTION + CONTEXT using:

  * Retrieved policies
  * Past decisions (memory)
    
* Enforces strict JSON schema:

```
{
  "decision": "Allowed | Not Allowed | Needs Review",
  "reason": "...",
  "suggested_changes": ["..."],
  "references": ["Policy Name Section X.Y"]
}
```

### 💾 **3. Memory Module**

Stores:

* Action
* Context
* Final LLM decision
  And feeds past 8 decisions back into the LLM.

### 🏗️ **4. LangGraph Workflow**

```
┌──────────┐     ┌────────────┐     ┌─────────┐
│ RETRIEVE │ →→  │  EVALUATE  │ →→  │ MEMORY │ →→ END
└──────────┘     └────────────┘     └─────────┘
```

### 🤖 **5. Full Local LLM Support**

* Loads any HuggingFace causal LM
* If loading fails → uses **dummy fallback LLM** with JSON output
* Ensures uninterrupted workflow

---

# 📂 Project File Structure

```
ASSIGNMENT_1/
│
├── data/
│   ├── data_policy.txt             # Policy document used for RAG
│   ├── fais_index_policy/          # Auto-generated FAISS index
│  
│── fais_index_policy/              # Auto-generated FAISS index
├── memory/
│   ├── memory_node.py              # Memory archiving node
│   └── state.py                    # AgentState Pydantic model
│
├── model/                          # Local model
│
├── output/
│   ├── sample1_output/
│   ├── sample2_output/
│   └── sample_input.txt
│
├── retriever.py                    # PolicyRetriever + LangGraph RAG node
├── governance_agent.py             # EvaluateActionNode (LLM evaluator)
├── graph.py                        # LangGraph workflow builder
├── main.py                         # CLI runner for the workflow
│
├── requirements.txt
├── README.md   
└── .gitignore
```

---

# ⚙️ How the System Works (Step-by-Step)

## **Step 1 — RETRIEVE**

`PolicyRetrieverNode.run()`

* Builds/loads FAISS index
* Retrieves the most relevant policy chunks
* Adds them to `state.retrieved_policies`

Logs:

```
🔍 Step 1: RETRIEVE - Searching for relevant policies...
Success: Retrieved 4 policy document(s).
```

---

## **Step 2 — EVALUATE**

`EvaluateActionNode.run()`

* Prepares the final prompt:

  * ACTION
  * CONTEXT
  * RETRIEVED POLICIES
  * PAST DECISIONS (memory)
  * JSON schema instructions
* Sends prompt to the LLM
* Parses output using `PydanticOutputParser`

Logs:

```
🧠 Step 2: EVALUATE - Running LLM for decision...
LLM Output Received (raw): { "decision": "Not Allowed", ... }
Decision: 'Not Allowed'
```

---

## **Step 3 — MEMORY**

`memory_node()`
Stores:

```json
{
  "action": "...",
  "context": "...",
  "result": {...}
}
```

Logs:

```
💾 Step 3: MEMORY - Archiving current decision...
Action 'Store user data...' archived.
Total decisions in memory: 1
```

## **Step 4 — OUTPUT**

`main.py` prints a clean JSON output:

```
===== GOVERNANCE DECISION (Clean Output) =====
{
  "decision": "Not Allowed",
  "reason": "...",
  "suggested_changes": [...],
  "references": [...]
}
=============================================
```

# 🚀 Running the Agent

```
CREATE ENV
ACTIVATE IT

```

## **1. Install Requirements**

```
pip install -r requirements.txt
```

## **2. Check model is available or not in model folder *

```
If no model is present → download it.
```

## **3. Run**

```
python main.py
```

# 📝 Input Format

You can enter:

### **Option 1: Raw JSON**

```
{
  "action": "Store user data on an analytics server",
  "context": "The team wants fast reporting"
}
```

Press enter twice to submit.

### **Option 2: Plain Action Text**


# 🧾 Sample Output

```
{
  "decision": "Not Allowed",
  "reason": "Storing user data on an external analytics server violates Data Storage Policy Section 1.1.",
  "suggested_changes": [
    "Obtain explicit approval from the Data Protection Officer.",
    "Encrypt user data before transmission."
  ],
  "references": [
    "Data Storage Policy Section 1.1"
  ]
}
```

# 🧠 Memory Example

After multiple queries:

```
PAST DECISIONS:
Past 1: action='Store data Externally...' | decision=Not Allowed
Past 2: action='Share logs...' | decision=Needs Review
...
```

The LLM now sees **historical behavior**, enabling more consistent governance decisions.

# 🛠️ Developer Notes

### 🔧 **FAISS index rebuilds automatically**

* When `data_policy.txt` is modified
* When index folder missing
* When timestamp mismatch detected

### 🧹 **Output is always clean JSON**
