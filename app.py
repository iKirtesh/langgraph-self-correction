import sys
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_code import compiled_code_graph, GOOGLE_API_KEY_PRESENT, GROQ_API_KEY_PRESENT
from agent_crag import compiled_crag_graph

app = Flask(__name__)
# Enable CORS for local cross-port routing
CORS(app)

# ==========================================
# CACHE-CONTROL HEADERS (Force Fresh Reloads)
# ==========================================
@app.after_request
def add_header(response):
    """
    Forces the browser to load the latest index.html templates and assets 
    by completely disabling client-side caching during development.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ==========================================
# UI RENDERING ROUTE
# ==========================================
@app.route('/', methods=['GET'])
def index():
    """
    Serves the premium, self-contained Self-Correction & Reflexion Arena dashboard.
    """
    return render_template('index.html')

# ==========================================
# REST API CONTROLLERS
# ==========================================
@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Exposes key configurations and credentials loaded on the backend.
    """
    return jsonify({
        "google_key_present": GOOGLE_API_KEY_PRESENT,
        "groq_key_present": GROQ_API_KEY_PRESENT,
        "mode": "Dynamic Self-Correction Sandbox"
    })

@app.route('/api/chat/code', methods=['POST'])
def post_chat_code():
    """
    Triggers the Code-Correction loop StateGraph, compiling code in a subprocess sandbox,
    feeding errors back to the LLM, and returning the full attempt audit trail.
    """
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    provider = data.get("provider", "gemini").strip()
    
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
        
    input_state = {
        "prompt": prompt,
        "code": "",
        "stdout": "",
        "stderr": "",
        "iteration": 0,
        "max_iterations": 4,
        "is_successful": False,
        "llm_provider": provider,
        "feedback": "",
        "attempts": []
    }
    
    try:
        # Run graph to completion
        final_state = compiled_code_graph.invoke(input_state)
        
        return jsonify({
            "status": "completed",
            "prompt": final_state.get("prompt"),
            "code": final_state.get("code"),
            "stdout": final_state.get("stdout"),
            "stderr": final_state.get("stderr"),
            "is_successful": final_state.get("is_successful"),
            "attempts": final_state.get("attempts")
        })
    except Exception as e:
        return jsonify({"error": f"Code-Correction loop execution failed: {str(e)}"}), 500

@app.route('/api/chat/crag', methods=['POST'])
def post_chat_crag():
    """
    Triggers the Corrective RAG (CRAG) & Self-RAG loop StateGraph, retrieves docs,
    rates relevance using evaluator nodes, calls web searches, and checks for hallucinations.
    """
    data = request.json or {}
    query = data.get("query", "").strip()
    provider = data.get("provider", "gemini").strip()
    
    if not query:
        return jsonify({"error": "Query is required."}), 400
        
    input_state = {
        "query": query,
        "documents": [],
        "graded_documents": [],
        "web_search_triggered": False,
        "web_results": "",
        "generation": "",
        "hallucination_score": "grounded",
        "llm_provider": provider,
        "steps": []
    }
    
    try:
        # Run graph to completion
        final_state = compiled_crag_graph.invoke(input_state)
        
        return jsonify({
            "status": "completed",
            "query": final_state.get("query"),
            "documents": final_state.get("documents"),
            "graded_documents": final_state.get("graded_documents"),
            "web_search_triggered": final_state.get("web_search_triggered"),
            "web_results": final_state.get("web_results"),
            "generation": final_state.get("generation"),
            "hallucination_score": final_state.get("hallucination_score"),
            "steps": final_state.get("steps")
        })
    except Exception as e:
        return jsonify({"error": f"Corrective RAG loop execution failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Force standard output UTF-8 encoding
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        
    # Launch Server on Port 5001 to avoid HITL conflicts
    app.run(host='0.0.0.0', port=5001, debug=True)
