import base64
import argparse
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BRANDS_FILE = DATA_DIR / "brands.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

load_dotenv(BASE_DIR / ".env")

DEFAULT_BRANDS = [
    "CeraVe",
    "Cetaphil",
    "La Roche-Posay",
    "The Ordinary",
    "Neutrogena",
]
DEFAULT_MEMORY = {"messages": [], "latest_analysis": None}
MEMORY_FILE = Path(
    os.environ.get(
        "MEMORY_FILE",
        Path(tempfile.gettempdir()) / "chat_memory.json"
        if os.environ.get("VERCEL")
        else DATA_DIR / "chat_memory.json",
    )
)
MAX_STORED_MEMORY_ITEMS = int(os.environ.get("MAX_STORED_MEMORY_ITEMS", "2000"))
MAX_MEMORY_MESSAGES = int(os.environ.get("MAX_MEMORY_MESSAGES", "60"))
MAX_MEMORY_CHARS = int(os.environ.get("MAX_MEMORY_CHARS", "12000"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-for-local-dev")
memory_lock = threading.Lock()


def ensure_data_file():
    try:
        DATA_DIR.mkdir(exist_ok=True)
    except OSError:
        return
    if not BRANDS_FILE.exists():
        try:
            BRANDS_FILE.write_text(json.dumps(DEFAULT_BRANDS, indent=2), encoding="utf-8")
        except OSError:
            pass


def ensure_memory_file():
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MEMORY_FILE.exists():
            MEMORY_FILE.write_text(json.dumps(DEFAULT_MEMORY, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_brands():
    ensure_data_file()
    try:
        return json.loads(BRANDS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_BRANDS


def save_brands(brands):
    ensure_data_file()
    normalized = sorted({brand.strip() for brand in brands if brand.strip()})
    BRANDS_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def load_memory():
    ensure_memory_file()
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = DEFAULT_MEMORY
    return {
        "messages": data.get("messages") if isinstance(data.get("messages"), list) else [],
        "latest_analysis": data.get("latest_analysis")
        if isinstance(data.get("latest_analysis"), dict)
        else None,
    }


def save_memory(memory):
    messages = memory.get("messages") if isinstance(memory.get("messages"), list) else []
    memory["messages"] = messages[-MAX_STORED_MEMORY_ITEMS:]
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def append_memory(role, content):
    if not content:
        return
    with memory_lock:
        memory = load_memory()
        memory["messages"].append(
            {
                "role": role,
                "content": str(content),
                "created_at": utc_now(),
            }
        )
        save_memory(memory)


def remember_analysis(result):
    with memory_lock:
        memory = load_memory()
        memory["latest_analysis"] = {
            "created_at": utc_now(),
            "result": result,
        }
        save_memory(memory)


def memory_text(entry):
    return str(entry.get("content", "")).strip()


def select_memory_messages(history, current_message):
    current_terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]{4,}", current_message)
        if len(term) >= 4
    }
    recent = history[-MAX_MEMORY_MESSAGES:]
    older = history[:-MAX_MEMORY_MESSAGES]
    relevant_older = [
        item
        for item in older
        if current_terms and current_terms.intersection(memory_text(item).lower().split())
    ]
    selected = (relevant_older[-12:] + recent)[-MAX_MEMORY_MESSAGES:]

    messages = []
    used_chars = 0
    for item in reversed(selected):
        content = memory_text(item)
        role = item.get("role")
        if role not in {"user", "assistant"} or not content:
            continue
        next_chars = used_chars + len(content)
        if next_chars > MAX_MEMORY_CHARS:
            break
        messages.append({"role": role, "content": content})
        used_chars = next_chars
    return list(reversed(messages))


def latest_analysis_context(latest_analysis):
    if not latest_analysis:
        return ""
    result = latest_analysis.get("result") or {}
    concerns = result.get("concerns") if isinstance(result.get("concerns"), list) else []
    routine = result.get("routine") if isinstance(result.get("routine"), list) else []
    routine_text = "; ".join(
        f"{item.get('step', 'Step')}: {item.get('recommendation', '')}"
        for item in routine
        if isinstance(item, dict)
    )
    return (
        "Latest saved skin scan context: "
        f"skin_type={result.get('skin_type', '-')}; "
        f"sensitivity={result.get('sensitivity', '-')}; "
        f"concerns={', '.join(map(str, concerns)) or '-'}; "
        f"routine={routine_text or '-'}; "
        f"notes={result.get('notes', '')}"
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_data_url(file_storage):
    mime_type = file_storage.mimetype or "image/png"
    image_bytes = file_storage.read()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_data_url(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("ascii")


def fallback_result(message):
    return {
        "ai_available": False,
        "skin_type": "-",
        "skin_information": [],
        "concerns": [],
        "sensitivity": "-",
        "psl_score": "-",
        "facial_rating": "-",
        "image_ratio_score": "-",
        "proportion_notes": [],
        "routine": [],
        "care_plan": [],
        "professional_summary": "",
        "notes": message,
    }


def build_analysis_prompt(brands):
    brand_text = ", ".join(brands) if brands else "No brands have been approved yet"
    return f"""
You are an educational skincare assistant, not a medical diagnostic tool.
Analyze the uploaded face image only for visible, non-sensitive skincare observations.
Do not identify the person, infer protected traits, diagnose disease, or promise results.
Only recommend products from this admin-approved brand list: {brand_text}.
Be concise. Do not write filler, generic paragraphs, or long explanations.
Routine and care advice must include only what is necessary from the uploaded image.

Return strict JSON with this schema:
{{
  "skin_type": "one of: oily, dry, combination, normal, unclear",
  "skin_information": ["2 to 4 short image-based observations about visible texture, oiliness/dryness, pores, tone, hydration, or clarity"],
  "concerns": ["short visible skincare concerns, or unclear"],
  "sensitivity": "skin quality from the image, such as clear, decent, uneven, dull, irritated-looking, acne-prone, dry-looking, oily-looking, or unclear",
  "psl_score": "face structure summary from the image, such as oval balanced, round soft features, angular jawline, long face, or unclear",
  "facial_rating": "brief visible facial structure note, or unclear",
  "image_ratio_score": "photo quality rating out of 10, such as 7/10, based on lighting, angle, sharpness, and face visibility",
  "proportion_notes": ["1 to 3 short notes about face structure, lighting, angle, and photo quality"],
  "routine": [
    {{"step": "Cleanse", "recommendation": "one short necessary recommendation"}},
    {{"step": "Moisturize", "recommendation": "one short necessary recommendation"}},
    {{"step": "Protect", "recommendation": "one short necessary recommendation"}},
    {{"step": "Treat", "recommendation": "one short optional recommendation only if visible concerns need it"}}
  ],
  "care_plan": ["2 to 4 short necessary care tips based on the uploaded image only"],
  "professional_summary": "one short plain-language summary of the visible skin condition",
  "notes": "one short educational disclaimer and when to see a dermatologist"
}}
"""


def parse_json_result(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    return normalize_result(parsed, ai_available=True)


def normalize_result(result, ai_available):
    return {
        "ai_available": ai_available,
        "skin_type": result.get("skin_type", "-"),
        "skin_information": result.get("skin_information")
        if isinstance(result.get("skin_information"), list)
        else [],
        "concerns": result.get("concerns") if isinstance(result.get("concerns"), list) else [],
        "sensitivity": result.get("sensitivity", "-"),
        "psl_score": result.get("psl_score", "-"),
        "facial_rating": result.get("facial_rating", "-"),
        "image_ratio_score": result.get("image_ratio_score", "-"),
        "proportion_notes": result.get("proportion_notes")
        if isinstance(result.get("proportion_notes"), list)
        else [],
        "routine": result.get("routine") if isinstance(result.get("routine"), list) else [],
        "care_plan": result.get("care_plan") if isinstance(result.get("care_plan"), list) else [],
        "professional_summary": result.get("professional_summary", ""),
        "notes": result.get("notes", ""),
    }


def analyze_with_openai(image_data_url, brands):
    from openai import OpenAI

    client = OpenAI()
    prompt = build_analysis_prompt(brands)
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )
    return parse_json_result(response.output_text)


def analyze_with_ollama(image_base64, brands):
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"
    model = os.environ.get("OLLAMA_MODEL", "gemma3")
    prompt = build_analysis_prompt(brands)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI service returned HTTP {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the AI service. Check your connection and restart the app."
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"AI service returned no message content: {data}")
    return parse_json_result(content)


def chat_with_ollama(message):
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"
    model = os.environ.get("OLLAMA_CHAT_MODEL", os.environ.get("OLLAMA_MODEL", "gemma3:12b"))
    memory = load_memory()
    history_messages = select_memory_messages(memory["messages"], message)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a very brief AI skincare assistant. Reply in 1 to 2 short sentences. "
                "Use the saved chat memory and latest scan context when helpful, but do not expose raw memory unless asked. "
                "Avoid long lists unless the user asks. Give educational skincare guidance, "
                "product routine help, and app usage help. Do not diagnose medical conditions. "
                "Tell users to see a dermatologist for urgent, painful, spreading, bleeding, "
                "infected, or persistent symptoms."
            ),
        }
    ]
    analysis_context = latest_analysis_context(memory.get("latest_analysis"))
    if analysis_context:
        messages.append({"role": "system", "content": analysis_context})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": message})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI service returned HTTP {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach the AI service.") from exc

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"AI service returned no chat content: {data}")
    return content


def configured_provider():
    provider = os.environ.get("AI_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return ""


def ai_status_label():
    provider = configured_provider()
    if provider == "ollama":
        return "AI ready"
    if provider == "openai":
        return "AI ready"
    return "AI key missing"


@app.get("/")
def index():
    return render_template(
        "index.html",
        brands=load_brands(),
        ai_configured=bool(configured_provider()),
        ai_status=ai_status_label(),
    )


@app.post("/admin/login")
def admin_login():
    password = request.json.get("password", "")
    expected = os.environ.get("ADMIN_PASSWORD", "admin123")
    if password != expected:
        return jsonify({"ok": False, "message": "Wrong admin password"}), 401
    session["admin"] = True
    return jsonify({"ok": True, "message": "Logged in"})


@app.post("/admin/brands")
def add_brand():
    if not session.get("admin"):
        return jsonify({"ok": False, "message": "Admin login required"}), 403
    brand = request.json.get("brand", "").strip()
    if not brand:
        return jsonify({"ok": False, "message": "Brand name is required"}), 400
    brands = load_brands()
    brands.append(brand)
    save_brands(brands)
    return jsonify({"ok": True, "brands": load_brands()})


@app.delete("/admin/brands/<brand>")
def delete_brand(brand):
    if not session.get("admin"):
        return jsonify({"ok": False, "message": "Admin login required"}), 403
    brands = [item for item in load_brands() if item.lower() != brand.lower()]
    save_brands(brands)
    return jsonify({"ok": True, "brands": load_brands()})


@app.post("/analyze")
def analyze():
    image = request.files.get("image")
    if not image or image.filename == "":
        return jsonify({"ok": False, "message": "Upload a face image first"}), 400
    if not allowed_file(image.filename):
        return jsonify({"ok": False, "message": "Use PNG, JPG, JPEG, or WEBP"}), 400

    image_bytes = image.read()
    provider = configured_provider()
    if not provider:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(
                    "AI is not configured. Set AI_PROVIDER and the matching API key, restart the app, and analyze again."
                ),
            }
        )

    try:
        if provider == "ollama":
            result = analyze_with_ollama(image_bytes_to_base64(image_bytes), load_brands())
        elif provider == "openai":
            mime_type = image.mimetype or "image/png"
            result = analyze_with_openai(image_bytes_to_data_url(image_bytes, mime_type), load_brands())
        else:
            return jsonify(
                {
                    "ok": True,
                    "result": fallback_result(f"Unsupported AI_PROVIDER: {provider}"),
                }
            )
    except Exception as exc:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(f"AI request failed: {exc}"),
            }
        )

    remember_analysis(result)
    return jsonify({"ok": True, "result": result})


@app.post("/chat")
def chat():
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify({"ok": False, "message": "Type a message first"}), 400

    provider = configured_provider()
    if provider != "ollama":
        return jsonify(
            {
                "ok": False,
                "message": "Chat is not configured.",
            }
        ), 400

    try:
        reply = chat_with_ollama(message)
        append_memory("user", message)
        append_memory("assistant", reply)
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Chat failed: {exc}"}), 500

    return jsonify({"ok": True, "reply": reply})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Dermatologist Assistant")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ensure_data_file()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=args.debug)
