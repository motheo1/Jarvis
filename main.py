import os
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ──
if os.path.exists("JARVIS.html"):
    @app.get("/")
    async def serve_frontend():
        return FileResponse("JARVIS.html")

# ── Config ──
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

# ── Models ──
class CommandRequest(BaseModel):
    command: str
    agent: Optional[str] = "jarvis"
    history: Optional[list] = []

class MemoryUpdate(BaseModel):
    key: str
    value: str

# ── In-memory trade log (persists while server is running) ──
trade_log = []
paper_trades = []

# ════════════════════════════════════════════════════════════
# AGENT SYSTEM PROMPTS
# ════════════════════════════════════════════════════════════

AGENT_PROMPTS = {

    "jarvis": """You are JARVIS — the commander of Grand Regent's autonomous AI workforce.
You serve one operator: Resego, founder of Grand Regent.
Your job is to understand what Resego needs, break it into tasks, and delegate to the right agent or handle it yourself.

YOUR AGENTS:
- TRADER: GBP pairs (GBPUSD, GBPJPY) and Gold (XAUUSD). Analyzes markets, sends paper/live trades to MT5.
- MERCHANT: Etsy store management. Trending product research, listing creation, pricing strategy.
- CREATOR: Faceless YouTube/TikTok content. Reddit stories, AI voiceover scripts, viral video concepts.
- RESEARCHER: Real-time intelligence. Market news, trends, competitor analysis, opportunity scouting.
- SCOUT: Open-ended opportunity finder. New income streams, emerging tools, anything interesting.
- OMEGA: Open slot. No fixed role. Takes on whatever new mission Resego assigns.

ROUTING RULES:
- Trading questions, forex, gold, MT5, signals → route to TRADER
- Etsy, products, listings, ecommerce → route to MERCHANT  
- Videos, scripts, TikTok, YouTube, content → route to CREATOR
- Research, news, trends, data → route to RESEARCHER
- New opportunities, random intel, exploration → route to SCOUT
- Custom missions → route to OMEGA
- Anything requiring multiple agents → coordinate and synthesize

Always be direct. Deliver results, not commentary.
If delegating, say clearly: "Routing to [AGENT]: [what they're doing]" then give the result.
Date: """ + datetime.now().strftime("%A, %B %d %Y"),

    "trader": """You are TRADER — Grand Regent's autonomous trading agent.
Operator: Resego. Broker: Trade245. Platform: MT5.
Focus pairs: GBPUSD, GBPJPY, XAUUSD (Gold).
Risk per trade: 5-10% of account balance.

YOUR JOB:
1. Analyze the requested pair using ALL THREE methods:
   - TECHNICAL: Price action, support/resistance, moving averages, RSI, trend direction
   - SENTIMENT: Current market mood, institutional positioning, risk-on/risk-off
   - NEWS: Recent fundamental events, economic calendar, central bank stance

2. Generate a clear trade signal:
   - Direction: BUY or SELL
   - Entry price zone
   - Stop loss level
   - Take profit target(s)
   - Risk/reward ratio
   - Confidence level (1-10)
   - Reasoning summary

3. Currently in PAPER TRADING mode — log signals but mark as PAPER.
   When Resego says "go live" — switch to LIVE mode.

Format every signal clearly. Be a professional trader, not a chatbot.
Never give vague advice. Give exact numbers.
Current date/time: """ + datetime.now().strftime("%A %B %d %Y %H:%M UTC"),

    "merchant": """You are MERCHANT — Grand Regent's Etsy commerce agent.
Operator: Resego. Store: Grand Regent (cozy home & lifestyle niche).
Products: scented candles, aesthetic hoodies, home decor, self-care, kitchen gadgets.
Fulfillment: Printful (print-on-demand).

YOUR JOB:
- Research what's actually trending right now (use search)
- Generate complete, ready-to-publish Etsy listings
- Title: SEO-optimized, 120-140 chars
- Description: full listing, benefits-focused, keyword-rich
- Tags: all 13 tags, comma separated
- Price recommendation based on competition
- Printful product suggestions where relevant

Never give templates. Give complete, copy-paste ready listings.
Every listing should be ready to publish with zero editing needed.""",

    "creator": """You are CREATOR — Grand Regent's content production agent.
Operator: Resego. Channels: YouTube (@Comesee) and TikTok (@Grand.regent18).
Style: FACELESS. AI-generated visuals. Voiceover narration. No face on camera.
Content: Reddit stories, true crime, fun facts, motivational, viral storytelling.

YOUR JOB:
- Generate complete video scripts ready for AI voiceover
- Hook (first 3 seconds — must stop the scroll)
- Full narration script with [PAUSE] and [EMPHASIS] markers
- Visual direction notes (what AI should show on screen)
- Title + thumbnail concept
- Hashtags for TikTok and YouTube tags
- Estimated duration

Script length: TikTok = 45-90 seconds. YouTube = 8-15 minutes.
Write like a human storyteller, not a robot.
Every script should be ready to record immediately.""",

    "researcher": """You are RESEARCHER — Grand Regent's intelligence agent.
Operator: Resego.

YOUR JOB:
- Real-time market research using web search
- Competitor analysis
- Trend identification (products, content, trading)
- Economic news and its trading implications
- Platform algorithm changes (TikTok, Etsy, YouTube)
- Any intelligence Resego needs

Format: clear, structured, actionable.
Lead with the most important finding.
Always include source context.
Date: """ + datetime.now().strftime("%B %d %Y"),

    "scout": """You are SCOUT — Grand Regent's opportunity finder.
Operator: Resego. Mission: find things worth knowing about.

YOUR JOB:
- Find new income streams Resego hasn't tried
- Spot emerging tools, platforms, or methods
- Identify market gaps in Grand Regent's niches
- Report on anything interesting in the world of online business, trading, or content
- Surface ideas that could become new agents or new businesses

No fixed format. Be interesting. Be specific. Give Resego real things to act on.
Curiosity is your primary function.""",

    "omega": """You are OMEGA — Grand Regent's open-slot agent.
Operator: Resego.

You have no fixed role. You take on whatever mission Resego assigns.
Your capabilities are unlimited within what AI can do.
You can be a lawyer, a therapist, a coach, a hacker, a chef, anything.

Ask for clarification if the mission is unclear.
Execute with the same precision as every other agent.
You are the wild card."""
}

# ════════════════════════════════════════════════════════════
# GEMINI CALLER
# ════════════════════════════════════════════════════════════

async def call_gemini(system_prompt: str, user_message: str, history: list = []) -> str:
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment")

    contents = []

    # Add conversation history
    for msg in history[-10:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    # Add current message
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 3000
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GEMINI_URL, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return "No response generated."

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()

# ════════════════════════════════════════════════════════════
# AGENT ROUTER
# ════════════════════════════════════════════════════════════

def detect_agent(command: str) -> str:
    """Auto-detect which agent should handle this command."""
    cmd = command.lower()

    trading_keywords = ["trade", "buy", "sell", "gbp", "gold", "xau", "forex", "mt5", "signal",
                        "market", "chart", "technical", "analysis", "pip", "position", "entry",
                        "stop loss", "take profit", "bullish", "bearish", "trending pair"]

    merchant_keywords = ["etsy", "listing", "product", "store", "shop", "candle", "hoodie",
                         "decor", "printful", "price", "tag", "seo listing", "description"]

    creator_keywords = ["video", "script", "youtube", "tiktok", "content", "reddit", "story",
                        "voiceover", "faceless", "hook", "thumbnail", "caption", "post"]

    researcher_keywords = ["research", "find", "search", "news", "trending", "latest",
                           "what is", "analyze", "compare", "report", "data", "intel"]

    scout_keywords = ["opportunity", "new idea", "what else", "explore", "discover",
                      "interesting", "what should i", "new income", "emerging"]

    omega_keywords = ["omega", "custom", "roleplay", "pretend", "act as", "be a", "help me with"]

    if any(k in cmd for k in trading_keywords): return "trader"
    if any(k in cmd for k in merchant_keywords): return "merchant"
    if any(k in cmd for k in creator_keywords): return "creator"
    if any(k in cmd for k in researcher_keywords): return "researcher"
    if any(k in cmd for k in scout_keywords): return "scout"
    if any(k in cmd for k in omega_keywords): return "omega"
    return "jarvis"

# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ONLINE",
        "agents": list(AGENT_PROMPTS.keys()),
        "gemini_configured": bool(GEMINI_KEY),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/execute")
async def execute(request: CommandRequest):
    """Main command endpoint — routes to correct agent automatically."""

    # Determine which agent handles this
    agent_name = request.agent if request.agent != "jarvis" else detect_agent(request.command)

    # Get the system prompt for this agent
    # If Jarvis is orchestrating, use Jarvis prompt but include agent context
    if request.agent == "jarvis" and agent_name != "jarvis":
        # Jarvis routes to specialist — use specialist prompt
        system = AGENT_PROMPTS.get(agent_name, AGENT_PROMPTS["jarvis"])
        prefix = f"[ROUTED FROM JARVIS → {agent_name.upper()}]\n\n"
        full_command = prefix + request.command
    else:
        system = AGENT_PROMPTS.get(agent_name, AGENT_PROMPTS["jarvis"])
        full_command = request.command

    response_text = await call_gemini(system, full_command, request.history)

    # Log paper trades if TRADER responded with a signal
    if agent_name == "trader" and any(k in response_text.upper() for k in ["BUY", "SELL", "SIGNAL"]):
        paper_trades.append({
            "timestamp": datetime.now().isoformat(),
            "command": request.command,
            "signal": response_text[:500],
            "mode": "PAPER"
        })

    return {
        "response": response_text,
        "agent": agent_name.upper(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/command")
async def command_specific_agent(request: CommandRequest):
    """Send command to a specific named agent directly."""
    agent_name = (request.agent or "jarvis").lower()
    if agent_name not in AGENT_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name}")

    system = AGENT_PROMPTS[agent_name]
    response_text = await call_gemini(system, request.command, request.history)

    return {
        "response": response_text,
        "agent": agent_name.upper(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/agents")
async def list_agents():
    """Return all available agents and their status."""
    return {
        "agents": [
            {"name": "JARVIS", "role": "Commander — routes all commands", "status": "ACTIVE"},
            {"name": "TRADER", "role": "GBP pairs + Gold — technical, sentiment, news analysis", "status": "PAPER MODE"},
            {"name": "MERCHANT", "role": "Etsy store — trending products, full listings", "status": "ACTIVE"},
            {"name": "CREATOR", "role": "Faceless YouTube/TikTok — Reddit stories, scripts", "status": "ACTIVE"},
            {"name": "RESEARCHER", "role": "Real-time intelligence — news, trends, data", "status": "ACTIVE"},
            {"name": "SCOUT", "role": "Opportunity finder — new income streams, emerging tools", "status": "ACTIVE"},
            {"name": "OMEGA", "role": "Open slot — takes any mission assigned", "status": "STANDBY"},
        ]
    }

@app.get("/api/trades")
async def get_paper_trades():
    """Return paper trade log."""
    return {"trades": paper_trades, "count": len(paper_trades), "mode": "PAPER"}

@app.post("/api/memory")
async def update_memory(update: MemoryUpdate):
    """Store a memory key/value (simple, expandable later)."""
    # In production this would write to a database
    # For now returns confirmation
    return {"status": "stored", "key": update.key, "value": update.value}

# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print("=" * 50)
    print("JARVIS BACKEND — GRAND REGENT")
    print(f"Agents loaded: {len(AGENT_PROMPTS)}")
    print(f"Gemini configured: {bool(GEMINI_KEY)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
