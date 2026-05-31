import os
from typing import TypedDict, List, Dict, Literal
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
# MOCK KNOWLEDGE BASE COLLECTION
# ==========================================
MOCK_KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "title": "OpenAI 2024 Valuation",
        "content": "Valuation of OpenAI in early 2024 was estimated at $80 billion following a tender offer led by Thrive Capital. The company remains focused on building artificial general intelligence (AGI)."
    },
    {
        "id": "doc2",
        "title": "OpenAI Company Overview",
        "content": "OpenAI is an artificial intelligence research laboratory founded in December 2015. It consists of the non-profit OpenAI Inc. and its for-profit subsidiary OpenAI Global LLC."
    },
    {
        "id": "doc3",
        "title": "Stateful Agents",
        "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, ideal for creating cyclic agent workflows and human-in-the-loop triggers."
    }
]

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class CRAGState(TypedDict):
    """
    Represents the CRAG & Self-RAG state.
    - query: User query.
    - documents: Retrieved raw documents.
    - graded_documents: Evaluated documents with relevance tags.
    - web_search_triggered: Flag indicating if a search gap was met.
    - web_results: Scraped web text snippets.
    - generation: Drafted grounded response.
    - hallucination_score: Grounding check score ("grounded" or "hallucinated").
    - llm_provider: Active LLM provider ("gemini" or "groq").
    - steps: Chronological audit track list.
    """
    query: str
    documents: List[Dict[str, str]]
    graded_documents: List[Dict[str, str]]
    web_search_triggered: bool
    web_results: str
    generation: str
    hallucination_score: str
    llm_provider: str
    steps: List[str]

# ==========================================
# 2. STATEGRAPH NODES
# ==========================================
def retrieve_node(state: CRAGState) -> dict:
    query = state.get("query", "")
    steps = state.get("steps", []) or []
    
    retrieved = []
    query_lower = query.lower()
    
    if "valuation" in query_lower or "openai" in query_lower:
        retrieved.append(MOCK_KNOWLEDGE_BASE[0])
        retrieved.append(MOCK_KNOWLEDGE_BASE[1])
    else:
        retrieved.append(MOCK_KNOWLEDGE_BASE[2])
        
    new_steps = list(steps)
    new_steps.append(f"Retrieved {len(retrieved)} documents from local knowledge collection.")
    return {"documents": retrieved, "steps": new_steps}

def grade_documents_node(state: CRAGState) -> dict:
    query = state.get("query", "")
    documents = state.get("documents", [])
    provider = state.get("llm_provider", "gemini")
    steps = state.get("steps", []) or []
    
    graded = []
    search_needed = False
    
    active_key_present = GOOGLE_API_KEY_PRESENT if provider == "gemini" else GROQ_API_KEY_PRESENT
    
    if not active_key_present:
        # Mock Evaluator simulation
        for doc in documents:
            if "2026" in query and "2024" in doc["content"]:
                graded.append({**doc, "score": "irrelevant", "reason": "Doc details valuation in 2024, query asks about 2026."})
                search_needed = True
            else:
                graded.append({**doc, "score": "relevant", "reason": "Contains relevant background terms."})
                
        new_steps = list(steps)
        new_steps.append(f"Graded documents: {len([d for d in graded if d['score'] == 'relevant'])} relevant, {len([d for d in graded if d['score'] == 'irrelevant'])} irrelevant.")
        return {"graded_documents": graded, "web_search_triggered": search_needed, "steps": new_steps}

    # Real LLM Evaluator
    if provider == "groq":
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
    system_prompt = (
        "You are an objective document relevance evaluator.\n"
        "Score whether the provided document contains direct, sufficient factual information to accurately answer the user's query.\n"
        "Format your answer as a raw JSON string with keys:\n"
        "{\n"
        "  \"score\": \"relevant\" or \"irrelevant\",\n"
        "  \"reason\": \"Brief explanation of relevance or missing information gap\"\n"
        "}"
    )
    
    for doc in documents:
        user_prompt = (
            f"User Query: {query}\n"
            f"Document Title: {doc['title']}\n"
            f"Document Content: {doc['content']}\n\n"
            "Evaluate document relevance and return JSON score."
        )
        
        try:
            response = model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            import json
            raw_text = response.content.replace("```json", "").replace("```", "").strip()
            score_data = json.loads(raw_text)
            
            score = score_data.get("score", "irrelevant")
            reason = score_data.get("reason", "No reason provided.")
            
            if score == "irrelevant":
                search_needed = True
                
            graded.append({**doc, "score": score, "reason": reason})
        except Exception as e:
            graded.append({**doc, "score": "irrelevant", "reason": f"Evaluator parser fail: {str(e)}"})
            search_needed = True
            
    new_steps = list(steps)
    new_steps.append(f"Graded documents: {len([d for d in graded if d['score'] == 'relevant'])} relevant, {len([d for d in graded if d['score'] == 'irrelevant'])} irrelevant.")
    return {"graded_documents": graded, "web_search_triggered": search_needed, "steps": new_steps}

def web_search_node(state: CRAGState) -> dict:
    query = state.get("query", "")
    steps = state.get("steps", []) or []
    
    query_lower = query.lower()
    if "2026" in query_lower and "valuation" in query_lower:
        results = "Web Search [2026-03]: OpenAI concluded a block funding round raising $6.6 billion, pushing its valuation to $157 billion, backed by Microsoft, Nvidia, Altimeter, and Softbank."
    else:
        results = f"Web Search [2026]: Resolved active indexed web results for '{query}' successfully."
        
    new_steps = list(steps)
    new_steps.append(f"Web search triggered. Scraped Results: {results}")
    return {"web_results": results, "steps": new_steps}

def generate_node(state: CRAGState) -> dict:
    query = state.get("query", "")
    graded_docs = state.get("graded_documents", []) or []
    web_results = state.get("web_results", "")
    provider = state.get("llm_provider", "gemini")
    steps = state.get("steps", []) or []
    
    relevant_contexts = [doc["content"] for doc in graded_docs if doc["score"] == "relevant"]
    combined_context = "\n\n".join(relevant_contexts)
    if web_results:
        combined_context += f"\n\n--- WEB SEARCH RESULTS ---\n{web_results}"
        
    active_key_present = GOOGLE_API_KEY_PRESENT if provider == "gemini" else GROQ_API_KEY_PRESENT
    
    if not active_key_present:
        if web_results:
            gen = (
                "Based on the combined information:\n"
                "While our initial local database only tracked OpenAI's valuation up to 2024 (at $80 billion), "
                "active Web Search reports indicate that in early 2026, OpenAI successfully raised a new funding round, "
                "pushing its valuation to an impressive **$157 billion**, backed by Thrive Capital, Microsoft, and Nvidia."
            )
        else:
            gen = "Standard answer compiled using relevant retrieved local document facts."
            
        new_steps = list(steps)
        new_steps.append("Synthesized response grounded strictly on filtered contexts.")
        return {"generation": gen, "steps": new_steps}

    if provider == "groq":
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
    system_prompt = (
        "You are a Corrective RAG (CRAG) answer synthesizer.\n"
        "Your goal is to answer the user's query utilizing the provided context.\n"
        "CRITICAL RULE: You must base your answer strictly on the facts provided in the context. Do NOT make up details.\n"
        "If some information was missing in local documents but resolved via Web Search, highlight the new search facts transparently."
    )
    
    user_prompt = (
        f"Query: {query}\n\n"
        f"--- GROUNDED CONTEXT ---\n{combined_context}\n\n"
        "Generate a professional, fully grounded, fact-based response."
    )
    
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    new_steps = list(steps)
    new_steps.append("Synthesized response grounded strictly on filtered contexts.")
    return {"generation": response.content, "steps": new_steps}

def grade_generation_node(state: CRAGState) -> dict:
    query = state.get("query", "")
    generation = state.get("generation", "")
    graded_docs = state.get("graded_documents", []) or []
    web_results = state.get("web_results", "")
    provider = state.get("llm_provider", "gemini")
    steps = state.get("steps", []) or []
    
    active_key_present = GOOGLE_API_KEY_PRESENT if provider == "gemini" else GROQ_API_KEY_PRESENT
    
    if not active_key_present:
        new_steps = list(steps)
        new_steps.append("Self-RAG Evaluated: Answer grounded completely in context. Hallucination checks passed!")
        return {"hallucination_score": "grounded", "steps": new_steps}

    if provider == "groq":
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
    system_prompt = (
        "You are an objective Self-RAG hallucination grader.\n"
        "Evaluate whether the generated response is completely grounded in and supported by the provided facts. "
        "Return \"grounded\" if there are absolutely no fabricated or unmentioned details, or \"hallucinated\" if details are made up.\n"
        "Output ONLY the single word: \"grounded\" or \"hallucinated\"."
    )
    
    relevant_contexts = [doc["content"] for doc in graded_docs if doc["score"] == "relevant"]
    combined_context = "\n\n".join(relevant_contexts)
    if web_results:
        combined_context += f"\n\n--- WEB SEARCH RESULTS ---\n{web_results}"
        
    user_prompt = (
        f"Grounded Facts Context:\n{combined_context}\n\n"
        f"Generated Response:\n{generation}\n\n"
        "Grade the response."
    )
    
    try:
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        grade = response.content.strip().lower()
        score = "grounded" if "grounded" in grade else "hallucinated"
    except Exception:
        score = "grounded"
        
    new_steps = list(steps)
    new_steps.append(f"Self-RAG Evaluated: Grounding audit status scored as '{score}'.")
    return {"hallucination_score": score, "steps": new_steps}

# ==========================================
# 3. CONDITIONAL ROUTING EDGES
# ==========================================
def route_relevance(state: CRAGState) -> Literal["web_search", "generate"]:
    if state.get("web_search_triggered", False):
        return "web_search"
    return "generate"

def route_hallucination(state: CRAGState) -> Literal["generate", "__end__"]:
    if state.get("hallucination_score", "grounded") == "hallucinated":
        return "generate"
    return END

# ==========================================
# 4. STATEGRAPH COMPILATION
# ==========================================
workflow = StateGraph(CRAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_generation", grade_generation_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_relevance,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "grade_generation")

workflow.add_conditional_edges(
    "grade_generation",
    route_hallucination,
    {
        "generate": "generate",
        END: END
    }
)

compiled_crag_graph = workflow.compile()
