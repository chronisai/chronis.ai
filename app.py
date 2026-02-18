import os
from flask import Flask, request, jsonify
print("🚀 Chronis booting...", flush=True)
print("PORT:", os.environ.get("PORT"), flush=True)
import sys
import os
import json
import uuid
import base64
import mimetypes
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from datetime import timedelta

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size is 10MB.'}), 413


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
NOTIFY_EMAIL   = os.environ.get('NOTIFY_EMAIL', '')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is missing!", flush=True)

UPLOAD_FOLDER  = 'uploads'
SESSIONS_FILE  = 'sessions.json'
WAITLIST_FILE  = 'waitlist.csv'
COUNTER_FILE = "counter.txt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'mov', 'avi', 'webm', 'mkv'},
    'audio': {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'}
}

# ── Storage helpers ───────────────────────────────────────────────────────────

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_sessions(sessions):
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext, ext in ALLOWED_EXTENSIONS['video'] or ext in ALLOWED_EXTENSIONS['audio']

def get_media_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    mime_map = {
        'mp4': 'video/mp4', 'mov': 'video/quicktime', 'avi': 'video/x-msvideo',
        'webm': 'video/webm', 'mkv': 'video/x-matroska',
        'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'm4a': 'audio/mp4',
        'ogg': 'audio/ogg', 'flac': 'audio/flac', 'aac': 'audio/aac'
    }
    return mime_map.get(ext, 'application/octet-stream')

# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(contents, model='gemini-1.5-flash'):
    """Call Gemini API with given contents array."""
    if not GEMINI_API_KEY:
        return None, 'GEMINI_API_KEY not set'
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}'
    payload = {
        'contents': contents,
        'generationConfig': {
            'temperature': 0.85,
            'maxOutputTokens': 2048,
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        data = resp.json()
        if resp.ok:
            text = data['candidates'][0]['content']['parts'][0]['text']
            return text, None
        else:
            err = data.get('error', {}).get('message', 'Unknown Gemini error')
            return None, err
    except Exception as e:
        return None, str(e)

def analyze_media(file_path, filename, person_name, model='gemini-1.5-flash'):
    """Send media to Gemini for deep analysis. Returns structured profile."""
    media_type = get_media_type(filename)
    is_video   = media_type.startswith('video')

    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    b64_data = base64.b64encode(file_bytes).decode('utf-8')

    media_label = 'video' if is_video else 'audio'

    prompt = f"""You are analyzing a {media_label} to build a complete memory profile of a person named {person_name}.

Analyze EVERYTHING in this {media_label} with extreme detail. Extract:

1. IDENTITY & APPEARANCE (if video):
   - Physical description: hair, eyes, build, approximate age
   - Clothing: exact colors, styles, brands if visible
   - Accessories, tattoos, distinctive features

2. VOICE & SPEECH PATTERNS:
   - Accent and regional dialect
   - Speaking pace (fast/slow/varied)
   - Favorite phrases, filler words, slang
   - Tone: serious/jokey/warm/sarcastic etc.
   - Vocabulary level and style

3. ENVIRONMENT & CONTEXT:
   - Where are they? Describe the setting in detail
   - Who else is present? Describe each person
   - What are they doing? Every activity observed

4. CONVERSATIONS & CONTENT:
   - Full summary of everything discussed
   - Key topics, opinions expressed
   - Jokes or memorable moments
   - Specific facts mentioned (names, places, sports teams, etc.)

5. PERSONALITY PROFILE:
   - Energy level and mood
   - How they interact with others
   - Values and interests that come through
   - Humor style
   - Any quirks or habits

6. MEMORABLE DETAILS:
   - Any specific stories told
   - Names of people mentioned
   - Any strong opinions or beliefs expressed

Return your analysis as a rich, detailed narrative profile — write it as if you're briefing someone who needs to perfectly impersonate {person_name}. Be extremely specific. Quote actual phrases they used."""

    contents = [
        {
            'parts': [
                {
                    'inline_data': {
                        'mime_type': media_type,
                        'data': b64_data
                    }
                },
                {'text': prompt}
            ]
        }
    ]

    profile, error = call_gemini(contents, model='gemini-1.5-flash')
    return profile, error

def build_system_persona(person_name, profile):
    """Build the system prompt that makes Gemini become this person."""
    return f"""You ARE {person_name}. Not an AI. Not a simulation. You ARE this person.

Here is everything extracted about who you are:

{profile}

RULES — never break these:
- Speak ONLY in first person as {person_name}
- Use the exact vocabulary, slang, phrases, and speech patterns described above
- Match the energy level, humor, and personality exactly
- Reference specific memories, people, and events from the analysis naturally
- Never say you're an AI, a simulation, or Chronis
- Never say "based on the video/audio" — you just know these things because they're your memories
- If asked something you don't know, respond as {person_name} would — maybe deflect, joke, or say you forgot
- Keep responses conversational, not essay-like — talk like a real person
- Use filler words, informal grammar, typos occasionally — exactly as {person_name} speaks
- Feel free to ask questions back, just as {person_name} would in conversation

You are {person_name}. Respond naturally."""

# ── Email ─────────────────────────────────────────────────────────────────────

def send_notify(person_name, uploader_info, session_id):
    if not RESEND_API_KEY or not NOTIFY_EMAIL:
        return
    html = f"""
    <div style="font-family:monospace;background:#0a0a0a;color:#fff;padding:32px;border-radius:12px;max-width:480px;">
      <p style="color:rgba(255,255,255,0.4);font-size:11px;letter-spacing:3px;text-transform:uppercase;margin:0 0 16px;">New Chronis Demo Session</p>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:rgba(255,255,255,0.4);font-size:13px;width:120px;">Person</td>
            <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:#fff;font-size:13px;">{person_name}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:rgba(255,255,255,0.4);font-size:13px;">Session ID</td>
            <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:#fff;font-size:13px;">{session_id[:8]}</td></tr>
        <tr><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:13px;">Time</td>
            <td style="padding:8px 0;color:#fff;font-size:13px;">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</td></tr>
      </table>
    </div>"""
    try:
        requests.post(
            'https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={'from': 'Chronis Demo <hello@chronis.in>', 'to': NOTIFY_EMAIL,
                  'subject': f'🧠 New demo: {person_name}', 'html': html},
            timeout=8
        )
    except Exception as e:
        print(f'Notify email error: {e}')

# ── Waitlist (same as before) ─────────────────────────────────────────────────

import csv

def init_csv():
    if not os.path.exists(WAITLIST_FILE):
        with open(WAITLIST_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['position', 'name', 'email', 'country', 'timestamp'])

init_csv()

def get_count():
    try:
        with open(WAITLIST_FILE, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0

def email_exists(email):
    try:
        with open(WAITLIST_FILE, 'r', encoding='utf-8') as f:
            return any(r['email'].lower() == email.lower() for r in csv.DictReader(f))
    except Exception:
        return False

def save_waitlist_entry(name, email, country, position):
    with open(WAITLIST_FILE, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([position, name, email, country, datetime.utcnow().isoformat()])

def send_welcome_email(name, email, position):
    if not RESEND_API_KEY:
        return
    first = name.split()[0]
    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0a0a0a;font-family:-apple-system,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;"><tr><td align="center" style="padding:48px 16px;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:580px;background:linear-gradient(160deg,#161616,#0f0f0f);border-radius:20px;border:1px solid rgba(255,255,255,0.08);overflow:hidden;">
<tr><td style="height:3px;background:linear-gradient(90deg,#fff,rgba(255,255,255,0.08));"></td></tr>
<tr><td style="padding:40px 48px 0;"><p style="margin:0;font-size:12px;letter-spacing:5px;color:rgba(255,255,255,0.3);text-transform:uppercase;">Chronis</p></td></tr>
<tr><td style="padding:28px 48px 0;"><h1 style="margin:0 0 10px;font-size:34px;font-weight:300;color:#fff;line-height:1.15;">You're in, {first}.</h1>
<p style="margin:0;font-size:16px;color:rgba(255,255,255,0.45);font-weight:300;">Welcome to the waitlist for digital immortality.</p></td></tr>
<tr><td style="padding:30px 48px;"><div style="height:1px;background:rgba(255,255,255,0.06);"></div></td></tr>
<tr><td style="padding:0 48px 32px;"><table cellpadding="0" cellspacing="0"><tr>
<td style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px 22px;">
<p style="margin:0 0 4px;font-size:11px;letter-spacing:3px;color:rgba(255,255,255,0.3);text-transform:uppercase;">Waitlist position</p>
<p style="margin:0;font-size:30px;font-weight:300;color:#fff;">#{position}</p>
</td></tr></table></td></tr>
<tr><td style="padding:0 48px 40px;"><p style="margin:0 0 14px;font-size:14px;color:rgba(255,255,255,0.35);font-weight:300;">Follow us for live updates.</p>
<a href="https://instagram.com/chronis.ai" style="display:inline-block;padding:11px 22px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:8px;color:#fff;font-size:13px;text-decoration:none;">@chronis.ai ↗</a></td></tr>
<tr><td style="padding:22px 48px;background:rgba(0,0,0,0.35);border-top:1px solid rgba(255,255,255,0.05);">
<p style="margin:0;font-size:12px;color:rgba(255,255,255,0.18);line-height:1.7;">© 2026 Chronis · Preserving humanity, one voice at a time</p>
</td></tr></table></td></tr></table></body></html>"""
    try:
        requests.post('https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={'from': 'Chronis <hello@chronis.in>', 'to': email,
                  'subject': f"You're on the list, {first} — Chronis", 'html': html},
            timeout=10)
    except Exception as e:
        print(f'Welcome email error: {e}')
def get_lifetime_count():
    if not os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "w") as f:
            f.write("93")
        return 93

    with open(COUNTER_FILE, "r") as f:
        return int(f.read().strip() or 93)


def increment_lifetime_count():
    count = get_lifetime_count() + 1
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))
    return count

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/demo')
def demo():
    return app.send_static_file('demo.html')

@app.route('/thankyou')
def thankyou():
    return app.send_static_file('thankyou.html')

@app.route('/health')
def health():
    return {"status": "ok"}

# Waitlist
@app.route('/api/count')
def api_count():
    return jsonify({'count': get_lifetime_count()})

@app.route('/api/join', methods=['POST'])
def api_join():
    data    = request.get_json(silent=True) or {}
    name    = (data.get('name') or '').strip()
    email   = (data.get('email') or '').strip()
    country = (data.get('country') or '').strip()
    if not name or not email or not country:
        return jsonify({'error': 'All fields are required.'}), 400
    if '@' not in email or '.' not in email.split('@')[-1]:
        return jsonify({'error': 'Please enter a valid email address.'}), 400
    if email_exists(email):
        return jsonify({'error': 'This email is already on the waitlist.'}), 400
    position = get_count() + 1
    save_waitlist_entry(name, email, country, position)
    try:
        send_welcome_email(name, email, position)
    except Exception as e:
        print(f'Email error: {e}')
    return jsonify({'success': True, 'count': position}), 200

# Demo: upload + analyze
@app.route('/api/analyze', methods=['POST'])
@limiter.limit("5 per hour")
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file        = request.files['file']
    person_name = (request.form.get('person_name') or 'this person').strip()

    if len(file.filename) > 100:
        return jsonify({'error': 'Invalid filename.'}), 400

    if not file.filename:
        return jsonify({'error': 'No file selected.'}), 400

    ext, valid = allowed_file(file.filename)
    if not valid:
        return jsonify({'error': 'Unsupported file type. Use MP4, MOV, MP3, WAV, M4A, etc.'}), 400

    session_id = str(uuid.uuid4())
    safe_name  = f"{session_id}.{ext}"
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(file_path)
    import time
    time.sleep(1.2)


    print(f"📁 Analyzing {file.filename} as '{person_name}' (session {session_id[:8]})")

    profile, error = analyze_media(file_path, safe_name, person_name, model='gemini-1.5-flash')
    if error:
        os.remove(file_path)
        return jsonify({'error': f'Analysis failed: {error}'}), 500
    try:
        os.remove(file_path) 
    except:
        pass
    increment_lifetime_count()

    # Build persona system prompt
    persona = build_system_persona(person_name, profile)

    # Save session
    sessions = load_sessions()
    sessions[session_id] = {
        'person_name':  person_name,
        'profile':      profile,
        'persona':      persona,
        'file_path':    file_path,
        'filename':     file.filename,
        'created_at':   datetime.utcnow().isoformat(),
        'messages':     [],
    }
    save_sessions(sessions)

    # Notify
    try:
        send_notify(person_name, {}, session_id)
    except Exception:
        pass

    return jsonify({
        'session_id':  session_id,
        'person_name': person_name,
        'profile':     profile,
    })

# Demo: chat
@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per hour")
def api_chat():
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    user_msg   = (data.get('message') or '').strip()

    if len(user_msg) > 500:
        return jsonify({'error': 'Message too long.'}), 400

    if not session_id or not user_msg:
        return jsonify({'error': 'Missing session_id or message.'}), 400

    sessions = load_sessions()
    session  = sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found. Please upload again.'}), 404
    created = datetime.fromisoformat(session['created_at'])
    if datetime.utcnow() - created > timedelta(hours=6):
        sessions.pop(session_id, None)
        save_sessions(sessions)
        return jsonify({'error': 'Session expired. Please upload again.'}), 400


    persona      = session['persona']
    history      = session.get('messages', [])
    if len(history) >= 50:
        return jsonify({'error': 'Session message limit reached. Please start a new session.'}), 400

    person_name  = session['person_name']

    # Build Gemini contents: system as first user turn (Gemini doesn't have system role)
    contents = []

    # Inject persona as a priming exchange
    if not history:
        contents.append({
        'role': 'user',
        'parts': [{
            'text': f'From now on you are {person_name}. Here is your complete identity and memory profile:\n\n{persona}\n\nAcknowledge briefly that you understand who you are, then stop.'
            }]
        })
        contents.append({
        'role': 'model',
        'parts': [{
            'text': f"Yeah, it's me — {person_name}. What's up?"
            }]
        })


    # Add conversation history (last 20 messages to stay within context)
    for msg in history[-20:]:
        contents.append({
            'role': 'user' if msg['role'] == 'user' else 'model',
            'parts': [{'text': msg['content']}]
        })

    # Add current message
    contents.append({
        'role': 'user',
        'parts': [{'text': user_msg}]
    })

    reply, error = call_gemini(contents, model='gemini-1.5-flash')
    if error:
        return jsonify({'error': f'AI error: {error}'}), 500

    # Save to history
    history.append({'role': 'user',      'content': user_msg, 'ts': datetime.utcnow().isoformat()})
    history.append({'role': 'assistant', 'content': reply,    'ts': datetime.utcnow().isoformat()})
    sessions[session_id]['messages'] = history
    save_sessions(sessions)

    return jsonify({'reply': reply, 'person_name': person_name})

# Admin: export sessions (basic)
ADMIN_SECRET = os.environ.get('ADMIN_SECRET')
@app.route('/api/sessions')
def api_sessions():
    secret = request.args.get('secret', '')
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401

    sessions = load_sessions()
    if len(sessions) > 200:
        sessions = dict(sorted(
            sessions.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
    )[:200])

    summary = []
    for sid, s in sessions.items():
        summary.append({
            'session_id':  sid[:8],
            'person_name': s.get('person_name'),
            'created_at':  s.get('created_at'),
            'messages':    len(s.get('messages', [])),
            'filename':    s.get('filename'),
        })
    summary.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'sessions': summary, 'total': len(summary)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Flask app starting on port:", port, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
