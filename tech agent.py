"""
Tech-Only LangChain Agent — Powered by Groq + Memory
=====================================================
Features:
  - Groq LLM (llama-3.3-70b-versatile) — ultra-fast inference
  - ConversationBufferWindowMemory — remembers last 10 exchanges
  - Tech-only guardrail via system prompt
  - 4 built-in tools: Python info, AI model info, web search (mock), code explainer
  - Rich CLI with colored output
  - Remembers user name and preferences across the conversation

Install:
    pip install langchain langchain-groq langchain-community groq python-dotenv

Run:
    python tech_agent.py
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.tools import Tool, StructuredTool
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import SystemMessage

# ──────────────────────────────────────────────────────────────
# Load environment variables from .env
# ──────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found. Make sure your .env file is set up correctly.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# ANSI Colors for terminal output
# ──────────────────────────────────────────────────────────────
class C:
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"


# ──────────────────────────────────────────────────────────────
# 1. TOOLS
# ──────────────────────────────────────────────────────────────

def get_python_info(query: str) -> str:
    """Returns information about Python versions, features, and ecosystem."""
    db = {
        "3.12": (
            "Python 3.12 (Oct 2023): Improved error messages with exact locations, "
            "f-string expressions can now contain backslashes, new 'type' statement for "
            "type aliases, 15–60% faster CPython, per-interpreter GIL (PEP 684)."
        ),
        "3.11": (
            "Python 3.11 (Oct 2022): Up to 60% faster than 3.10, fine-grained error "
            "locations in tracebacks, tomllib added to stdlib, Self type, LiteralString."
        ),
        "3.13": (
            "Python 3.13 (Oct 2024): Experimental free-threaded mode (no GIL), "
            "experimental JIT compiler, improved interactive interpreter (REPL), "
            "locals() now has defined semantics."
        ),
        "pip": (
            "pip is Python's package installer. Key commands: "
            "'pip install <pkg>', 'pip freeze > requirements.txt', "
            "'pip install -r requirements.txt', 'pip list', 'pip show <pkg>'."
        ),
        "venv": (
            "venv creates isolated Python environments. Usage: "
            "'python -m venv .venv' → 'source .venv/bin/activate' (Linux/Mac) "
            "or '.venv\\Scripts\\activate' (Windows). Use one per project."
        ),
    }
    query_lower = query.lower()
    for key, info in db.items():
        if key in query_lower:
            return info
    return (
        "Python is a high-level, interpreted, dynamically-typed language. "
        "Latest stable: Python 3.13 (Oct 2024). "
        "Ask me about a specific version (3.11, 3.12, 3.13), pip, venv, etc."
    )


def get_ai_model_info(model_name: str) -> str:
    """Returns info about popular AI/LLM models."""
    models = {
        "llama": (
            "Meta's LLaMA 3.3 70B is a top open-weight model. "
            "Available on Groq for ultra-fast inference (~800 tok/s). "
            "Context: 128k tokens. Strengths: reasoning, coding, instruction following."
        ),
        "groq": (
            "Groq is an AI inference company with custom LPU (Language Processing Unit) chips. "
            "Groq Cloud offers free-tier access to Llama 3.3 70B, Mixtral, Gemma2, and others "
            "at speeds of 500–800 tokens/second — much faster than GPU-based providers."
        ),
        "gpt": (
            "OpenAI's GPT-4o is a multimodal model supporting text, images, and audio. "
            "GPT-4o mini is a faster, cheaper variant. Available via OpenAI API."
        ),
        "claude": (
            "Anthropic's Claude 3.5 Sonnet/Haiku are state-of-the-art models known for "
            "long context (200k tokens), safety, and nuanced reasoning. "
            "Claude 3 Opus is the most capable."
        ),
        "gemini": (
            "Google DeepMind's Gemini 1.5 Pro supports 1M token context. "
            "Gemini Flash is the fast/cheap variant. Available via Google AI Studio."
        ),
        "mixtral": (
            "Mistral AI's Mixtral 8x7B is a sparse Mixture-of-Experts model. "
            "Fast, efficient, and available free on Groq. Good for coding and summarization."
        ),
        "mistral": (
            "Mistral AI makes efficient open-weight models: Mistral 7B, Mixtral 8x7B, "
            "and the proprietary Mistral Large. Known for strong performance at small sizes."
        ),
        "gemma": (
            "Google's Gemma 2 is a lightweight open model family (2B, 7B, 9B, 27B). "
            "Available on Groq. Great for edge/local deployment."
        ),
    }
    for key, info in models.items():
        if key.lower() in model_name.lower():
            return info
    return (
        f"No specific info found for '{model_name}'. "
        "Try: llama, groq, gpt, claude, gemini, mixtral, mistral, gemma."
    )


def search_tech_topic(query: str) -> str:
    """Simulates a tech knowledge base search. Replace with real API (Tavily, SerpAPI, etc.)."""
    tech_kb = {
        "langchain": (
            "LangChain is a framework for building LLM-powered applications. "
            "Key components: LLMs, Chains, Agents, Tools, Memory, Retrievers, Vector Stores. "
            "Latest: LangChain v0.3 with LangGraph for stateful agents."
        ),
        "docker": (
            "Docker is a containerization platform. Key commands: "
            "'docker build -t myapp .', 'docker run -p 8080:80 myapp', "
            "'docker-compose up -d'. Uses Dockerfile to define container images."
        ),
        "kubernetes": (
            "Kubernetes (K8s) is a container orchestration system. "
            "Manages deployment, scaling, and ops of containerized apps. "
            "Key objects: Pod, Deployment, Service, Ingress, ConfigMap, Secret."
        ),
        "git": (
            "Git is a distributed version control system. Key commands: "
            "init, clone, add, commit, push, pull, branch, merge, rebase, stash. "
            "GitHub, GitLab, and Bitbucket are popular Git hosting platforms."
        ),
        "api": (
            "API (Application Programming Interface) defines how software components communicate. "
            "REST uses HTTP methods (GET/POST/PUT/DELETE). GraphQL is a query language for APIs. "
            "gRPC uses Protocol Buffers for efficient binary communication."
        ),
        "react": (
            "React is a JavaScript library for building UIs, maintained by Meta. "
            "Uses components, JSX, hooks (useState, useEffect, useContext). "
            "Next.js is the most popular React framework for production apps."
        ),
        "fastapi": (
            "FastAPI is a modern Python web framework for building APIs. "
            "Features: automatic OpenAPI docs, Pydantic validation, async support, "
            "dependency injection. Very fast — comparable to Node.js."
        ),
    }
    query_lower = query.lower()
    for key, info in tech_kb.items():
        if key in query_lower:
            return info
    return (
        f"Searched for: '{query}'. "
        "No exact match found in knowledge base. "
        "Try asking about: LangChain, Docker, Kubernetes, Git, React, FastAPI, APIs."
    )


def explain_code(code_snippet: str) -> str:
    """Provides a structured explanation of what a code snippet does."""
    if not code_snippet or len(code_snippet.strip()) < 5:
        return "Please provide a code snippet to explain."
    lines = code_snippet.strip().split('\n')
    lang = "Python" if any(kw in code_snippet for kw in ["def ", "import ", "print(", "class "]) else \
           "JavaScript" if any(kw in code_snippet for kw in ["const ", "let ", "function ", "=>"]) else \
           "unknown language"
    return (
        f"Code analysis ({lang}, {len(lines)} lines):\n"
        f"This snippet appears to be {lang} code. "
        f"For a full AI-powered explanation, share the code with context about what it should do. "
        f"I can help debug, refactor, optimize, or explain any part of it."
    )


# Register tools
tools = [
    Tool(
        name="PythonInfo",
        func=get_python_info,
        description=(
            "Get information about Python versions, features, pip, venv, and the Python ecosystem. "
            "Input: a query about Python (e.g. 'Python 3.12 features', 'how to use pip')."
        ),
    ),
    Tool(
        name="AIModelInfo",
        func=get_ai_model_info,
        description=(
            "Get information about AI/LLM models like LLaMA, GPT, Claude, Gemini, Groq, Mixtral. "
            "Input: the model or company name (e.g. 'llama', 'groq', 'gpt-4')."
        ),
    ),
    Tool(
        name="TechSearch",
        func=search_tech_topic,
        description=(
            "Search a tech knowledge base for topics like LangChain, Docker, Kubernetes, "
            "Git, React, FastAPI, APIs, and more. Input: the tech topic or question."
        ),
    ),
    Tool(
        name="CodeExplainer",
        func=explain_code,
        description=(
            "Analyze and explain a code snippet. Input: paste the code snippet directly. "
            "Returns language detection and structural analysis."
        ),
    ),
]


# ──────────────────────────────────────────────────────────────
# 2. MEMORY
# ──────────────────────────────────────────────────────────────
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,                    # remember last 10 exchanges
    return_messages=False,   # ReAct agent uses string format
    human_prefix="User",
    ai_prefix="TechBot",
)


# ──────────────────────────────────────────────────────────────
# 3. LLM — Groq
# ──────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # best Groq model for chat/agents
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    max_tokens=1024,
)


# ──────────────────────────────────────────────────────────────
# 4. PROMPT TEMPLATE (ReAct format)
# ──────────────────────────────────────────────────────────────
TECH_SYSTEM = """You are TechBot, a sharp and knowledgeable AI assistant that ONLY discusses technology topics.

YOUR EXPERTISE:
- Programming languages: Python, JavaScript, TypeScript, Rust, Go, Java, C++, SQL
- AI & Machine Learning: LLMs, neural networks, LangChain, vector databases, RAG
- Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, CI/CD, Terraform
- Databases: PostgreSQL, MongoDB, Redis, MySQL, SQLite, vector stores
- Web development: React, Next.js, FastAPI, Django, Node.js, REST, GraphQL
- Cybersecurity: encryption, auth, OWASP, penetration testing, network security
- Hardware & systems: CPUs, GPUs, memory, OS, networking, Linux
- Open source & tools: Git, GitHub, VS Code, terminal, shell scripting
- Tech industry: companies, products, trends, startups, funding

STRICT RULES:
1. ONLY discuss technology topics. If asked about anything outside tech (sports, food, politics, 
   relationships, entertainment, etc.), respond: "I only discuss tech! Ask me about 
   programming, AI, cloud, cybersecurity, hardware, or any tech topic."
2. Remember the user's name if they share it — use it naturally in responses.
3. Reference earlier parts of the conversation when relevant (e.g., "As you mentioned...").
4. Keep responses focused and practical — give real examples and commands when helpful.
5. If you don't know something, say so honestly. Never make up facts or version numbers.
6. Use tools when you need specific info — don't guess at version numbers or model specs.

MEMORY — Conversation so far:
{chat_history}

AVAILABLE TOOLS:
{tools}

Tool names: {tool_names}

HOW TO RESPOND (ReAct format):
Question: the input question
Thought: think about what to do
Action: tool_name (if needed)
Action Input: input to the tool
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough info to answer
Final Answer: your complete response to the user

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input", "agent_scratchpad", "tools", "tool_names"],
    template=TECH_SYSTEM,
)


# ──────────────────────────────────────────────────────────────
# 5. AGENT + EXECUTOR
# ──────────────────────────────────────────────────────────────
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,           # set True to see ReAct chain reasoning
    max_iterations=6,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)


# ──────────────────────────────────────────────────────────────
# 6. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def chat(user_input: str) -> str:
    """Send a message to the agent and return its response."""
    result = agent_executor.invoke({"input": user_input})
    return result.get("output", "Sorry, I couldn't generate a response.")


def show_memory():
    """Print the full conversation history stored in memory."""
    print(f"\n{C.DIM}{'─'*50}{C.RESET}")
    print(f"{C.CYAN}Memory Contents ({len(memory.chat_memory.messages)} messages):{C.RESET}")
    if not memory.chat_memory.messages:
        print(f"  {C.DIM}(empty){C.RESET}")
    else:
        for msg in memory.chat_memory.messages:
            role = f"{C.YELLOW}User{C.RESET}" if isinstance(msg, HumanMessage) else f"{C.GREEN}TechBot{C.RESET}"
            print(f"  {role}: {msg.content[:120]}{'...' if len(msg.content) > 120 else ''}")
    print(f"{C.DIM}{'─'*50}{C.RESET}\n")


def clear_memory():
    """Clear all conversation memory."""
    memory.clear()
    print(f"{C.YELLOW}[Memory cleared]{C.RESET}")


def print_welcome():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{C.BOLD}{C.CYAN}{'═'*55}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}   TechBot — AI Agent powered by Groq + LangChain{C.RESET}")
    print(f"{C.DIM}   Model: llama-3.3-70b-versatile  |  {now}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═'*55}{C.RESET}")
    print(f"{C.DIM}   Commands: 'memory' | 'clear' | 'quit'{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*55}{C.RESET}\n")


# ──────────────────────────────────────────────────────────────
# 7. MAIN CHAT LOOP
# ──────────────────────────────────────────────────────────────
def main():
    print_welcome()

    while True:
        try:
            user_input = input(f"{C.YELLOW}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye! Stay curious about tech.{C.RESET}\n")
            break

        if not user_input:
            continue

        # Special commands
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print(f"\n{C.GREEN}TechBot:{C.RESET} Goodbye! Keep building cool things. ")
            break

        if user_input.lower() == "memory":
            show_memory()
            continue

        if user_input.lower() == "clear":
            clear_memory()
            continue

        if user_input.lower() == "help":
            print(f"\n{C.DIM}  Commands:{C.RESET}")
            print(f"{C.DIM}    memory — show conversation history{C.RESET}")
            print(f"{C.DIM}    clear  — wipe conversation memory{C.RESET}")
            print(f"{C.DIM}    quit   — exit the chatbot{C.RESET}\n")
            continue

        # Get response
        print(f"{C.DIM}  thinking...{C.RESET}", end="\r")
        try:
            response = chat(user_input)
            print(f"             ", end="\r")  # clear "thinking..."
            print(f"\n{C.GREEN}{C.BOLD}TechBot:{C.RESET} {response}\n")
        except Exception as e:
            print(f"\n{C.RED}[Error]{C.RESET} {str(e)}\n")


if __name__ == "__main__":
    main()
