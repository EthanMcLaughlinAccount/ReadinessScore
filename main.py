import os, time, uuid, math, json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import requests

# ---- Load XGBoost via skops (recommended format for HF/xgb) ----
from skops.io import load as skops_load

APP_NAME = "entrepreneur-readiness-api"
VERSION = "1.0.0"

# env
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MODEL_PATH      = os.getenv("MODEL_PATH", "model/model.skops")  # your exported xgb model here
MODEL_ID        = os.getenv("MODEL_ID", "ethnmcl/entrepreneur-readiness-xgb")
HF_API_KEY      = os.getenv("HF_API_KEY", "").strip()
HF_SUMMARY_MODEL= os.getenv("HF_SUMMARY_MODEL", "ethnmcl/gpt2-entrepreneur-agent")

# ---- App ----
app = FastAPI(title=APP_NAME, version=VERSION)

# CORS
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != [""] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Simple in-memory rate limiter (per IP) ----
from collections import defaultdict, deque
import time as _t
WINDOW_SEC = float(os.getenv("RL_WINDOW_SEC", "3.0"))
MAX_REQS   = int(os.getenv("RL_MAX_REQS", "15"))
_hits: Dict[str, deque] = defaultdict(deque)

def rate_limit(ip: str) -> bool:
    now = _t.time()
    dq = _hits[ip]
    while dq and now - dq[0] > WINDOW_SEC:
        dq.popleft()
    if len(dq) >= MAX_REQS:
        return False
    dq.append(now)
    return True

# ---- Logging / request id middleware ----
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()
    response: Response
    try:
        response = await call_next(request)
    except Exception as e:
        response = Response(status_code=500, content=json.dumps({
            "error": {"type":"internal_error","message":str(e),"request_id":rid}
        }), media_type="application/json")
    dur = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = rid
    response.headers["x-response-time-ms"] = str(dur)
    return response

# ---- Model loading ----
_model = None
def load_model():
    global _model
    if _model is None:
        _model = skops_load(MODEL_PATH)  # must be created from your trained xgb model
    return _model

# ---- Schemas ----
class InputPayload(BaseModel):
    savings: float
    monthly_income: float
    monthly_expenses: float
    monthly_entertainment: float
    sales_skills: float
    dependents: float
    age: float
    assets: float
    risk_level: float
    confidence: float
    idea_difficulty: float

    # optional/derived
    entrepreneurial_readiness_score: Optional[float] = Field(default=None)  # label; never used in predict
    savings_ratio: Optional[float] = None
    expense_ratio: Optional[float] = None
    entertainment_ratio: Optional[float] = None

    # optional strings (ignored by model; carried through for context)
    entrepreneur_type: Optional[str] = None
    education_level: Optional[str] = None
    funding_source: Optional[str] = None

    @validator("*", pre=True)
    def coerce(cls, v):
        # make "3" -> 3 where sensible
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return v
            try:
                if any(k in cls.__fields__ for k in []):  # placeholder
                    pass
            except: 
                pass
        return v

    def with_ratios(self) -> "InputPayload":
        d = self.dict()
        mi = float(d.get("monthly_income") or 0.0)
        if not d.get("savings_ratio"):
            d["savings_ratio"] = (float(d.get("savings") or 0.0) / (mi if mi != 0 else 1.0))
        if not d.get("expense_ratio"):
            d["expense_ratio"] = (float(d.get("monthly_expenses") or 0.0) / (mi if mi != 0 else 1.0))
        if not d.get("entertainment_ratio"):
            d["entertainment_ratio"] = (float(d.get("monthly_entertainment") or 0.0) / (mi if mi != 0 else 1.0))
        return InputPayload(**d)

class ScoreResponse(BaseModel):
    score: float
    band: str
    meta: Dict[str, Any]

class SummarizeRequest(BaseModel):
    input: InputPayload
    score: float
    band: str

class SummarizeResponse(BaseModel):
    summary: str
    meta: Dict[str, Any]

# ---- Utils ----
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def band(score: float) -> str:
    return "Low" if score < 50 else ("Med" if score < 75 else "High")

def normalize_score(y: float) -> float:
    try:
        v = float(y)
    except:
        v = 0.0
    return clamp(v * 100.0 if 0.0 <= v <= 1.0 else v, 0.0, 100.0)

def summary_prompt(inp: dict, score: float, band_str: str) -> str:
    return (
        "You are a concise analyst for an Entrepreneur Readiness Agent.\n\n"
        "GOAL\n"
        "Write a clear 2–4 sentence summary of the founder/startup based on structured inputs and a readiness score.\n"
        'End with ONE specific, actionable next step on a new line prefixed with "Next:".\n\n'
        "RULES\n"
        "- Do NOT invent metrics or claims not present in the input JSON.\n"
        "- Reflect the provided score and band exactly as given.\n"
        "- If data is missing/empty, acknowledge it briefly instead of guessing.\n\n"
        f"INPUT\nJSON: {json.dumps(inp, ensure_ascii=False)}\n"
        f"Score (0–100): {score}\nBand: {band_str}\n\nBEGIN SUMMARY\n"
    )

def hf_textgen(prompt: str) -> str:
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF_API_KEY not configured on server")
    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type":"application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.4,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    # simple retries on 429/503
    for attempt in range(3):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (429, 503):
            time.sleep((0.7 * (2 ** attempt)))
            continue
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF upstream error {r.status_code}: {r.text[:400]}")
        try:
            js = r.json()
            if isinstance(js, list) and js and isinstance(js[0], dict) and "generated_text" in js[0]:
                return js[0]["generated_text"].strip()
            return str(js).strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"HF parse error: {str(e)}")

# ---- Routes ----
@app.get("/healthz")
def healthz():
    return {"status":"ok","uptime_s": int(time.process_time())}

@app.post("/v1/score", response_model=ScoreResponse)
def score(req: Request, body: InputPayload):
    ip = req.client.host if req.client else "unknown"
    if not rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limited")
    rid = str(uuid.uuid4())
    t0 = time.time()

    # compute ratios & drop label for features
    body2 = body.with_ratios()
    features = body2.dict()
    features.pop("entrepreneurial_readiness_score", None)  # never send labels

    # order features if your model expects specific order (example keeps dict order)
    model = load_model()
    try:
        # If your skops model is a sklearn/xgb pipeline with predict_proba:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([list(features.values())])[0]
            # choose positive class prob if binary
            y = float(p[1]) if len(p) > 1 else float(p[0])
        else:
            # fallback: plain predict
            y = float(model.predict([list(features.values())])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    score_val = normalize_score(y)
    band_val  = band(score_val)

    dur = int((time.time() - t0) * 1000)
    return ScoreResponse(
        score=score_val,
        band=band_val,
        meta={"duration_ms": dur, "model_id": MODEL_ID, "request_id": rid}
    )

@app.post("/v1/summarize", response_model=SummarizeResponse)
def summarize(req: Request, body: SummarizeRequest):
    ip = req.client.host if req.client else "unknown"
    if not rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limited")
    rid = str(uuid.uuid4())
    t0 = time.time()

    # build the prompt using the *input exactly as provided*
    inp = body.input.with_ratios().dict()
    # never leak labels to external callers (not used here anyway, but keep clean)
    prompt = summary_prompt(inp, body.score, body.band)

    text = hf_textgen(prompt)
    dur = int((time.time() - t0) * 1000)
    return SummarizeResponse(
        summary=text,
        meta={"duration_ms": dur, "model_id": HF_SUMMARY_MODEL, "request_id": rid}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
