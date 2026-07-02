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

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
    timeout=60.0,
    max_retries=3,
)

llm_name = "openai/gpt-4.1-mini"

app = FastAPI(title="AI Agent API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


# ====================== AGENT (unchanged from original) ======================

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
        response = client.chat.completions.create(
            model=llm_name,
            temperature=0.0,
            messages=self.messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content


# ====================== TOOLS ======================

# Original: return eval(what, {"__builtins__": {}})
# That is still dangerous once this runs on the internet instead of on a
# personal machine -- {"__builtins__": {}} does NOT fully sandbox eval(), it
# can be escaped (e.g. via ().__class__.__base__.__subclasses__() chains) to
# run arbitrary code on the server. Same function name/signature, safe internals:

_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
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
    tree = ast.parse(what, mode="eval")
    return _safe_eval(tree.body)


def planet_mass(name):
    masses = {
        "Mercury": 0.33011, "Venus": 4.8675, "Earth": 5.972, "Mars": 0.64171,
        "Jupiter": 1898.19, "Saturn": 568.34, "Uranus": 86.813, "Neptune": 102.413,
    }
    name = name.strip().capitalize()
    if name not in masses:
        return f"Planet '{name}' not found."
    return f"{name} has a mass of {masses[name]} x 10^24 kg"


known_actions = {"calculate": calculate, "planet_mass": planet_mass}

# ====================== PROMPT (unchanged from original) ======================

prompt = """
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
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should find the mass of each planet using planet_mass.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 x 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 x 10^24 kg

Next, call the agent again with:

Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 0.64171 x 10^24 kg

You then output:

Answer: Mars has a mass of 0.64171 x 10^24 kg

Finally, calculate the combined mass.

Action: calculate: 5.972 + 0.64171
PAUSE

Observation: The combined mass is 6.61371 x 10^24 kg

Answer: The combined mass of Earth and Mars is 6.61371 x 10^24 kg
""".strip()

action_re = re.compile(r"^Action: (\w+): (.*)$")


# ====================== query() LOOP, adapted for the web ======================
# Same logic as the original query()/query_interactive() functions: run the
# agent, check for an Action line, execute the tool, feed the Observation
# back, repeat until there's no more action or max_turns is hit.

def run_agent(question, max_turns=10):
    bot = Agent(prompt)
    next_prompt = question
    steps = []
    result = ""

    for _ in range(max_turns):
        result = bot(next_prompt)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]

        if not actions:
            return result, steps

        action, action_input = actions[0].groups()
        if action not in known_actions:
            return result, steps

        try:
            observation = known_actions[action](action_input)
        except Exception as e:
            observation = f"Error running {action}: {e}"

        steps.append({"action": action, "input": action_input, "result": str(observation)})
        next_prompt = f"Observation: {observation}"

    return result, steps


# ====================== WEB ENDPOINTS ======================

class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@app.get("/")
async def root():
    return {"status": "online", "message": "AI Agent API is running on Railway!"}


@app.get("/api/test")
@limiter.limit("10/minute")
async def test(request: Request):
    return {"status": "success", "message": "API is working!", "tools": list(known_actions.keys())}


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(payload: ChatPayload, request: Request):
    try:
        result, steps = run_agent(payload.message)
        return {"response": result, "steps": steps}
    except Exception as e:
        logger.error(f"Model API error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Something went wrong talking to the AI model.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
