# REGULATION-AWARE RAG SYSTEM WITH GOVERNANCE AGENT

This project implements a Regulation-Aware Retrieval-Augmented Generation (RAG) System integrated with a Governance Agent.
Its objective is to automatically evaluate a proposed action against a set of policy or regulatory documents, determine its compliance status, provide a clear rationale referencing specific policies, and suggest necessary changes for alignment.

## FEATURES

1. Policy Ingestion & Indexing

   - Automatically loads a policy document.
   - Chunks it into fine-grained clauses.
   - Creates a high-efficiency FAISS index for semantic search.

2. Semantic Retrieval (RAG Component)

   - Uses a pre-trained SentenceTransformer model to perform semantic search.
   - Retrieves the most relevant policy clauses for a given user query.

3. Governance Decision Making

   - Employs a Gemma-2-2B language model as the Governance Agent.
   - Uses a structured prompt to generate a compliance decision:
     Allowed, Not Allowed, or Needs Review.

4. Structured Output

   - Enforces a strict JSON output format for the agent's decision.
   - Ensures consistency in the decision, reasoning, suggested changes, and policy references.

5. Interactive Workflow

   - Runs in an interactive loop.
   - Allows continuous evaluation of different proposed actions.

## WORKFLOW EXECUTION (STEP-BY-STEP)

PHASE 1: SYSTEM INITIALIZATION

- Load the SentenceTransformer model and policy document.
- Chunk the document and generate embeddings.
- Build the FAISS index.
- Load the Gemma-2-2B model and tokenizer.

PHASE 2: USER INPUT PROCESSING

- User submits a proposed action, for example:
  {"action": "Share customer emails with third-party marketing",
  "context": "New product launch campaign"}
- The system parses and extracts the ACTION and CONTEXT.

PHASE 3: POLICY RETRIEVAL (RAG COMPONENT)

- A query embedding is generated for the combined user input.
- A semantic search is performed on the FAISS index.
- The top-k relevant policy clauses are retrieved, scored, and formatted.

PHASE 4: GOVERNANCE DECISION MAKING

- A structured prompt is created, incorporating the user action, context, and retrieved clauses.
- The Gemma-2-2B agent generates a response.
- The system extracts and validates the structured JSON output.

PHASE 5: RESULT PRESENTATION

- Retrieved policy clauses (with text and scores) are displayed.
- The final structured decision from the Governance Agent is shown, highlighting:
  Decision, Reason, and Suggested Changes.

## QUICK START & INSTALLATION

**1. Installation Steps**

# Clone and setup environment

---git clone repository

---cd ASSIGNMENT_1

---python -m venv rag_env

---source rag_env/bin/activate  # Windows: rag_env\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Ensure model files are in ./model/ directory

# Run the main.py file

python main.py
