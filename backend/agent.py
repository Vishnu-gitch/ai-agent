import os
import re
import ast
import operator
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load .env (only used for local development; Railway injects real env vars)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API")

# ============================================
# RATE LIMITING (protects your API key from abuse)
# ============================================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================
# CORS - only allow your real frontend domains
# ============================================
# List every exact frontend URL you actually deploy to via env var.
# Wildcards like "https://*.vercel.app" do NOT work with allow_origins,
# so we use allow_origin_regex instead for Netlify preview URLs.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================
# GITHUB MODELS CLIENT
# ============================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    logger.warning("GITHUB_TOKEN not set! Set it as an environment variable, never in code.")

try:
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=GITHUB_TOKEN,
        timeout=60.0,
        max_retries=3,
    )
    logger.info("OpenAI-compatible client initialized")
except Exception as e:
    logger.error(f"Failed to initialize client: {e}")
    client = None

LLM_NAME = "openai/gpt-4.1-mini"
ACTION_RE = re.compile(r"^Action: (\w+): (.*)$")

# ============================================
# AGENT CLASS
# ============================================


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        if client is None:
            return "Error: AI client not initialized. Please check server configuration."
        try:
            response = client.chat.completions.create(
                model=LLM_NAME,
                temperature=0.0,
                messages=self.messages,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Model API error: {type(e).__name__}: {e}", exc_info=True)
            return "Error: something went wrong talking to the AI model."


# ============================================
# TOOL FUNCTIONS
# ============================================

# Safe calculator: parses the expression into an AST and only allows
# numbers and basic math operators. No eval(), no arbitrary code execution.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculate(what):
    try:
        if len(what) > 200:
            return "Calculation error: expression too long"
        tree = ast.parse(what, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def planet_mass(name):
    masses = {
        "Mercury": 0.33011, "Venus": 4.8675, "Earth": 5.972, "Mars": 0.64171,
        "Jupiter": 1898.19, "Saturn": 568.34, "Uranus": 86.813, "Neptune": 102.413,
    }
    cleaned = name.strip().capitalize()
    if cleaned not in masses:
        return f"Planet '{name}' not found."
    return f"{cleaned} has a mass of {masses[cleaned]} x 10^24 kg"


def sum_planet_masses(planet_names):
    masses = {
        "mercury": 0.33011, "venus": 4.8675, "earth": 5.972,
        "mars": 0.64171, "jupiter": 1898.19, "saturn": 568.34,
        "uranus": 86.813, "neptune": 102.413,
    }
    names = [name.strip().lower() for name in planet_names.split(',')]
    total = 0
    found = []
    missing = []
    for name in names:
        if name in masses:
            total += masses[name]
            found.append(name.capitalize())
        else:
            missing.append(name)
    if not found:
        return "No valid planets found."
    result = f"Total mass: {total} x 10^24 kg"
    if found:
        result += f"\nIncluded: {', '.join(found)}"
    if missing:
        result += f"\nNot found: {', '.join(missing)}"
    return result


KNOWN_ACTIONS = {
    "calculate": calculate,
    "planet_mass": planet_mass,
    "sum_masses": sum_planet_masses,
}

# ============================================
# PROMPT
# ============================================

PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

planet_mass:
e.g. planet_mass: Earth
Returns the mass of a planet in the solar system

sum_masses:
e.g. sum_masses: Earth, Mars
Returns the total mass of multiple planets

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should use sum_masses to get the total.
Action: sum_masses: Earth, Mars
PAUSE

Observation: Total mass: 6.61371 x 10^24 kg
Answer: The combined mass of Earth and Mars is 6.61371 x 10^24 kg.
""".strip()

# ============================================
# PYDANTIC MODEL (with input validation)
# ============================================


class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list = []


# ============================================
# ENDPOINTS
# ============================================


@app.get("/")
async def root():
    return {"status": "online", "message": "AI Agent API is running on Railway!"}


@app.get("/api/test")
@limiter.limit("10/minute")
async def test(request: Request):
    return {
        "status": "success",
        "message": "API is working!",
        "tools": list(KNOWN_ACTIONS.keys()),
    }


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(payload: ChatPayload, request: Request):
    try:
        logger.info(f"Received message of length {len(payload.message)}")

        if client is None:
            raise HTTPException(status_code=503, detail="AI service is not configured.")

        agent = Agent(system=PROMPT)
        result = agent(payload.message)

        max_turns = 5
        steps = []

        for turn in range(max_turns):
            actions = [ACTION_RE.match(a) for a in result.split("\n") if ACTION_RE.match(a)]
            if not actions:
                break

            action, action_input = actions[0].groups()
            if action not in KNOWN_ACTIONS:
                logger.warning(f"Unknown action requested: {action}")
                break

            observation = KNOWN_ACTIONS[action](action_input)
            steps.append({"action": action, "input": action_input, "result": observation})

            next_prompt = f"Observation: {observation}"
            result = agent(next_prompt)

        clean_result = result
        clean_result = re.sub(r'^Action:.*$\n?', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'^PAUSE$\n?', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'^Observation:.*$\n?', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'^Thought:.*$\n?', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'^Answer:\s*', '', clean_result, flags=re.MULTILINE)
        clean_result = clean_result.strip()

        return {
            "response": clean_result or result.strip() or "I couldn't generate a response.",
            "history": agent.messages,
            "steps": steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
