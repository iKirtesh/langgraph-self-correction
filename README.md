# 🦜🔗 Self-Correction & Reflexion Loops Arena

A state-of-the-art, full-stack application demonstrating **stateful self-reflection and self-repairing loops** in AI agent architectures. Powered by **LangGraph**, **Google Gemini 2.5**, **Groq Llama 3.3**, and **Flask**, this arena lets you witness agents auditing, grading, and repairing themselves dynamically!

---

## 💻 Subprocess Code-Correction Sandbox Loop

The agent generates Python code based on a prompt, executes it in an isolated sandbox, captures traceback exceptions, and feeds errors back into its model to patch bugs in real-time.

```
                  [ START ]
                      │
                      ▼
             ┌─────────────────┐
             │  generate_code  │◄──────────────┐
             └────────┬────────┘               │
                      │                        │
                      ▼                        │ (if error)
             ┌─────────────────┐               │
             │  execute_code   ├───────────────┤ reflect_and_fix
             └────────┬────────┘               │
                      │                        │
                      │ (success or limit)     │
                      ▼                        │
                   [ END ] ────────────────────┘
```

### Key Sandbox Features:
- **Isolated Compiler Subprocess**: Runs generated Python files inside a secure subprocess (`subprocess.run`) with strict `5-second timeouts` to prevent runaway executions.
- **Traceback Interception**: The sandbox captures all standard console outputs (`stdout`) and compile-time exception logs (`stderr`), including `NameError`, `IndexError`, and `SyntaxError`.
- **Iterative Reflection**: If the script crashes, the trace stack is fed back to the LLM node, which isolates the bug, applies corrections, and recompiles until a successful run is achieved (up to a 4-attempt limit guard).

---

## 🔍 Corrective RAG (CRAG) & Self-RAG

A robust context grading and grounding architecture that objective grades document relevance, triggers live web search helpers to patch gaps, and audits grounding to prevent hallucinations.

### Workflow Logic:
1. **retrieve**: Fetches relevant information snippets from a local vector mock collection.
2. **grade_documents**: An LLM grader node scores retrieved documents as `"relevant"` or `"irrelevant"`.
3. **web_search**: If a critical context gap is discovered (e.g. outdated valuation data), the agent automatically triggers a dynamic web search to pull fresh facts.
4. **generate**: Synthesizes a response strictly grounded on the combined context.
5. **grade_generation (Self-RAG)**: A grounding audit node scores whether the response has hallucinations. If hallucination is detected, it loops back to regenerate!

---

## 📂 Project Architecture

```
d:/Agentic AI/langgraph-self-correction/
├── pyproject.toml         # Requirements (Flask, LangGraph, LLM packages)
├── .gitignore             # Strict ignoring rules (excludes active .env credentials)
├── README.md              # Premium documentation
├── agent_code.py          # StateGraph compiler for the Code-Correction loop
├── agent_crag.py          # StateGraph for Corrective RAG and Self-RAG
├── app.py                 # Flask REST backend and endpoints (Port 5001)
└── templates/
    └── index.html         # Frosted glassmorphic playground dashboard
```

---

## 🛠️ Setup & Run Instructions

### Prerequisites
Ensure you have [Astral `uv`](https://github.com/astral-sh/uv) installed on your system.

### 1. Environment Configurations
Verify that you have created a `.env` file at the root folder specifying your active API credentials:
```env
GOOGLE_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=false
```

### 2. Boot the Server
Open a terminal in the project directory and launch the Flask server using `uv`:
```bash
uv run python app.py
```
*The `uv` package manager will automatically coordinate all virtual environment dependencies and sync libraries in under 3 seconds!*

### 3. Open Visual Dashboard
Visit **`http://127.0.0.1:5001`** inside your web browser.

---

## 💻 Manual Verification Guide

### Test Code-Correction Loop
1. Click the **Force Traceback Repair** prompt card in the playground dashboard.
2. Observe the state graph blink `generate_code` ➔ `execute_code`.
3. Read Attempt #1 in the compiler log to see the name error traceback: `NameError: name 'undef_var' is not defined`.
4. Watch as the agent automatically triggers the correction node, fixes the scope bug, and returns a successful run result in Attempt #2!

### Test Corrective RAG (CRAG)
1. Swap to **Corrective RAG** mode.
2. Click the **Valuation Info Gaps** card (`What is OpenAI's latest valuation in 2026?`).
3. Observe the evaluator grade local documents as irrelevant (valuation stops at 2024).
4. Watch as the agent blinks the `web_search` node, pulls fresh 2026 funding details ($157B), checks grounding, and prints a hallucination-free response!
