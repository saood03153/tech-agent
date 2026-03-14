# TechBot — AI Agent powered by Groq + LangChain

A conversational AI agent that **only discusses technology topics** and remembers your conversation history. Built with LangChain, Groq (LLaMA 3.3 70B), and a ReAct agent loop with 4 built-in tools.

---

## Features

- **Tech-only guardrail** — refuses non-tech topics politely
- **Conversation memory** — remembers last 10 exchanges (name, preferences, context)
- **4 built-in tools** — Python info, AI model info, tech search, code explainer
- **Groq LLM** — ultra-fast inference (~700 tokens/sec) using LLaMA 3.3 70B
- **ReAct agent loop** — thinks, picks tools, observes, then answers
- **Colored CLI** — clean terminal interface with special commands

---

## Project Structure

```
tech-chatbot-agent/
├── tech_agent.py        # Main agent — LLM + memory + tools + CLI
├── requirements.txt     # Python dependencies
├── .env                 # Your API keys (DO NOT commit this)
├── .gitignore           # Ignores .env and cache files
└── README.md            # This file
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/tech-chatbot-agent.git
cd tech-chatbot-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Click **API Keys** → **Create API Key**
4. Copy the key

### 4. Set up your `.env` file

```bash
cp .env.example .env
```

Then open `.env` and replace `your_groq_api_key_here` with your actual key.

### 5. Run the agent

```bash
python tech_agent.py
```

---

## Usage

```
You: hi, my name is Ahmed and I want to learn Python
TechBot: Hey Ahmed! Great choice — Python is one of the best languages to start with...

You: what are the best Python 3.12 features?
TechBot: [uses PythonInfo tool] Python 3.12 brought some great improvements...

You: what do you know about groq?
TechBot: [uses AIModelInfo tool] Groq is an AI inference company with custom LPU chips...
```

### Special commands

| Command  | Description                        |
|----------|------------------------------------|
| `memory` | Show full conversation history     |
| `clear`  | Wipe memory and start fresh        |
| `help`   | Show available commands            |
| `quit`   | Exit the chatbot                   |

---

## Built-in Tools

| Tool            | Description                                      |
|-----------------|--------------------------------------------------|
| `PythonInfo`    | Python versions, pip, venv, stdlib info          |
| `AIModelInfo`   | LLaMA, GPT, Claude, Gemini, Groq, Mixtral info  |
| `TechSearch`    | Knowledge base: LangChain, Docker, Git, React…  |
| `CodeExplainer` | Analyze and explain code snippets                |

---

## Configuration

In `tech_agent.py` you can change:

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # swap to "mixtral-8x7b-32768" or "gemma2-9b-it"
    temperature=0.3,                   # 0 = deterministic, 1 = creative
    max_tokens=1024,
)

memory = ConversationBufferWindowMemory(
    k=10,   # number of past exchanges to remember
)
```

### Available Groq models

| Model                      | Speed     | Best for              |
|----------------------------|-----------|-----------------------|
| `llama-3.3-70b-versatile`  | Fast      | Complex reasoning     |
| `mixtral-8x7b-32768`       | Very fast | Long context tasks    |
| `gemma2-9b-it`             | Fastest   | Simple Q&A            |

---

## Extending the Agent

To add a new tool, define a function and register it:

```python
def my_new_tool(query: str) -> str:
    # your logic here
    return "result"

tools.append(Tool(
    name="MyTool",
    func=my_new_tool,
    description="Describe when the agent should use this tool.",
))
```

To add real web search, install [Tavily](https://tavily.com) and add `TAVILY_API_KEY` to your `.env`.

---

## License

MIT — free to use, modify, and distribute.
