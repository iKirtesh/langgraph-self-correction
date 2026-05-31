import os
import sys
import re
import subprocess
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Check for API Keys
GOOGLE_API_KEY_PRESENT = bool(os.getenv("GOOGLE_API_KEY")) and os.getenv("GOOGLE_API_KEY") != "your_google_api_key_here"
GROQ_API_KEY_PRESENT = bool(os.getenv("GROQ_API_KEY")) and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here"

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class CodeState(TypedDict):
    """
    Represents the compiler loop state.
    - prompt: User requirements.
    - code: Compiled Python code script.
    - stdout: Printed output.
    - stderr: Tracebacks and compile syntax errors.
    - iteration: Active compiler attempt count.
    - max_iterations: Limit threshold guard.
    - is_successful: Sandbox validation status.
    - llm_provider: Active LLM backend ("gemini" or "groq").
    - feedback: Detailed error logs sent to correction nodes.
    - attempts: Complete historical audit list for visual logs.
    """
    prompt: str
    code: str
    stdout: str
    stderr: str
    iteration: int
    max_iterations: int
    is_successful: bool
    llm_provider: str
    feedback: str
    attempts: list

# ==========================================
# 2. UTILITY CODE PARSER
# ==========================================
def extract_python_code(text: str) -> str:
    """
    Safely extracts raw Python code from markdown code fences.
    """
    # Try custom python fence
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic fence
    pattern_generic = r"```(.*?)```"
    match_gen = re.search(pattern_generic, text, re.DOTALL)
    if match_gen:
        return match_gen.group(1).strip()
        
    return text.strip()

# ==========================================
# 3. NODE COMPONENT LOGIC
# ==========================================
def generate_code_node(state: CodeState) -> dict:
    prompt = state.get("prompt", "")
    iteration = state.get("iteration", 0)
    feedback = state.get("feedback", "")
    code = state.get("code", "")
    provider = state.get("llm_provider", "gemini")
    
    # 1. Check if keys are active
    active_key_present = GOOGLE_API_KEY_PRESENT if provider == "gemini" else GROQ_API_KEY_PRESENT
    
    if not active_key_present:
        # Simulation Mode
        if iteration == 0:
            simulated_code = (
                "def calculate_factorial(n):\n"
                "    # INTENTIONAL BUG: undef_var variable used without assignment\n"
                "    print(f'Attempting calculation inside sandbox. Flag: {undef_var}')\n"
                "    if n == 0 or n == 1:\n"
                "        return 1\n"
                "    return n * calculate_factorial(n - 1)\n\n"
                "print(f'Factorial result: {calculate_factorial(5)}')"
            )
            simulated_msg = "Simulating initial code generation with variable naming bug."
        else:
            simulated_code = (
                "def calculate_factorial(n):\n"
                "    # BUG FIXED: undef_var assigned correctly inside function context\n"
                "    undef_var = 'Self-Correction Sandbox Active'\n"
                "    print(f'Attempting calculation inside sandbox. Flag: {undef_var}')\n"
                "    if n == 0 or n == 1:\n"
                "        return 1\n"
                "    return n * calculate_factorial(n - 1)\n\n"
                "print(f'Factorial result: {calculate_factorial(5)}')"
            )
            simulated_msg = "Simulating error reflection and automatic patch generation."
            
        return {
            "code": simulated_code,
            "feedback": simulated_msg
        }

    # 2. Real LLM Mode
    if provider == "groq":
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
    system_prompt = (
        "You are an elite, sandboxed Python code generator.\n"
        "Your goal is to write clean, high-performance, and perfectly executable Python scripts.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY a valid Python code block enclosed in ```python\\n[code]\\n```.\n"
        "2. Do NOT write any introduction, summaries, descriptions, or comments outside the code block.\n"
        "3. Always write standard console outputs using `print(...)` so results can be evaluated by the execution sandbox."
    )
    
    if feedback:
        user_prompt = (
            f"Your previous Python script failed to execute. Analyze the compile traceback below, "
            f"identify the cause of failure, apply the correct fix, and generate the FULL corrected Python script.\n\n"
            f"--- FAILING CODE SCRIPTS ---\n{code}\n\n"
            f"--- COMPILER TRACEBACK ---\n{feedback}\n\n"
            f"Original Intended Goal: {prompt}\n\n"
            f"Write the complete, bug-free, repaired Python script."
        )
    else:
        user_prompt = (
            f"Goal: {prompt}\n\n"
            f"Write a fully functional, executable Python script to achieve this."
        )
        
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    extracted_code = extract_python_code(response.content)
    
    return {
        "code": extracted_code,
        "feedback": "New code version compiled."
    }

def execute_code_node(state: CodeState) -> dict:
    code = state.get("code", "")
    iteration = state.get("iteration", 0) + 1
    attempts = state.get("attempts", []) or []
    provider = state.get("llm_provider", "gemini")
    
    active_key_present = GOOGLE_API_KEY_PRESENT if provider == "gemini" else GROQ_API_KEY_PRESENT
    
    if not active_key_present:
        # Simulation compiler runs
        if "undef_var" in code:
            stdout = ""
            stderr = "NameError: name 'undef_var' is not defined on line 3"
            success = False
        else:
            stdout = "Attempting calculation inside sandbox. Flag: Self-Correction Sandbox Active\nFactorial result: 120"
            stderr = ""
            success = True
            
        new_attempts = list(attempts)
        new_attempts.append({
            "iteration": iteration,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "is_successful": success
        })
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "is_successful": success,
            "iteration": iteration,
            "attempts": new_attempts,
            "feedback": stderr if not success else "Execution Succeeded!"
        }

    # Sandboxed execution
    temp_file = "temp_compiler_sandbox.py"
    try:
        # Write to temporary module
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        # Run isolated execution environment
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        stdout = result.stdout
        stderr = result.stderr
        success = (result.returncode == 0)
        
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Compiler Timeout Error: Execution exceeded isolated sandbox limits (5 seconds)."
        success = False
    except Exception as e:
        stdout = ""
        stderr = f"Sandbox Infrastructure Error: {str(e)}"
        success = False
    finally:
        # Clean up sandbox script
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
    new_attempts = list(attempts)
    new_attempts.append({
        "iteration": iteration,
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
        "is_successful": success
    })
    
    return {
        "stdout": stdout,
        "stderr": stderr,
        "is_successful": success,
        "iteration": iteration,
        "attempts": new_attempts,
        "feedback": stderr if not success else "Execution Succeeded!"
    }

# ==========================================
# 4. CONDITIONAL ROUTING EDGE
# ==========================================
def should_continue(state: CodeState) -> Literal["generate_code", "__end__"]:
    if state.get("is_successful", False):
        return END
        
    if state.get("iteration", 0) >= state.get("max_iterations", 4):
        return END
        
    return "generate_code"

# ==========================================
# 5. GRAPH COMPILATION
# ==========================================
workflow = StateGraph(CodeState)

workflow.add_node("generate_code", generate_code_node)
workflow.add_node("execute_code", execute_code_node)

workflow.add_edge(START, "generate_code")
workflow.add_edge("generate_code", "execute_code")

workflow.add_conditional_edges(
    "execute_code",
    should_continue,
    {
        "generate_code": "generate_code",
        END: END
    }
)

compiled_code_graph = workflow.compile()
