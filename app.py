import os
import sys
import json
import uuid
import base64
import requests
import tempfile
import stripe
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

print("🚀 Chronis booting...", flush=True)

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get('GEMINI_API_KEY', '')
GROQ_API_KEY       = os.environ.get('GROQ_API_KEY', '')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')
RESEND_API_KEY     = os.environ.get('RESEND_API_KEY', '')
NOTIFY_EMAIL       = os.environ.get('NOTIFY_EMAIL', '')
ADMIN_SECRET       = os.environ.get('ADMIN_SECRET', '')
SUPABASE_URL       = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY       = os.environ.get('SUPABASE_SERVICE_KEY', '')
STRIPE_SECRET_KEY  = os.environ.get('STRIPE_SECRET_KEY', '')
SITE_URL           = os.environ.get('SITE_URL', 'https://chronis.in')
XTTS_SPACE_URL     = os.environ.get('XTTS_SPACE_URL', '')   # e.g. https://chronisai-chronis-tts.hf.space
XTTS_SECRET        = os.environ.get('XTTS_SECRET', '')       # matches API_SECRET in HF Space

FREE_MSG_LIMIT = 5  # free messages before paywall

stripe.api_key = STRIPE_SECRET_KEY

UPLOAD_FOLDER = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'mov', 'avi', 'webm', 'mkv'},
    'audio': {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'}
}

MIME_MAP = {
    'mp4': 'video/mp4', 'mov': 'video/quicktime', 'avi': 'video/x-msvideo',
    'webm': 'video/webm', 'mkv': 'video/x-matroska',
    'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'm4a': 'audio/mp4',
    'ogg': 'audio/ogg', 'flac': 'audio/flac', 'aac': 'audio/aac'
}

# ── Supabase helpers ──────────────────────────────────────────────────────────

def _sb_headers(extra=None):
    h = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }
    if extra:
        h.update(extra)
    return h

def sb_select(table, query=''):
    try:
        r = requests.get(f'{SUPABASE_URL}/rest/v1/{table}?{query}', headers=_sb_headers(), timeout=10)
        return r.json() if r.ok else []
    except Exception as e:
        print(f'[SB select error] {e}', flush=True)
        return []

def sb_count(table, query=''):
    try:
        r = requests.get(f'{SUPABASE_URL}/rest/v1/{table}?select=id&{query}', headers=_sb_headers({'Prefer': 'count=exact'}), timeout=10)
        return int(r.headers.get('Content-Range', '0/0').split('/')[-1])
    except Exception as e:
        print(f'[SB count error] {e}', flush=True)
        return 0

def sb_insert(table, data):
    try:
        r = requests.post(f'{SUPABASE_URL}/rest/v1/{table}', headers=_sb_headers(), json=data, timeout=10)
        rows = r.json()
        return rows[0] if r.ok and rows else None
    except Exception as e:
        print(f'[SB insert error] {e}', flush=True)
        return None

def sb_update(table, match_col, match_val, data):
    try:
        r = requests.patch(f'{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}', headers=_sb_headers(), json=data, timeout=10)
        return r.ok
    except Exception as e:
        print(f'[SB update error] {e}', flush=True)
        return False

# ── Waitlist helpers ──────────────────────────────────────────────────────────

WAITLIST_BASELINE = 93

def get_waitlist_count():
    return max(sb_count('waitlist') + WAITLIST_BASELINE, WAITLIST_BASELINE)

def email_exists(email):
    return len(sb_select('waitlist', f'email=eq.{requests.utils.quote(email)}&select=id')) > 0

def save_waitlist_entry(name, email, country, position):
    sb_insert('waitlist', {'name': name, 'email': email, 'country': country, 'position': position})

# ── Session helpers ───────────────────────────────────────────────────────────

def get_session(session_id):
    rows = sb_select('sessions', f'session_id=eq.{session_id}&select=*')
    return rows[0] if rows else None

def create_session(session_id, person_name, profile, persona, filename, voice_id=None):
    sb_insert('sessions', {
        'session_id':  session_id,
        'person_name': person_name,
        'profile':     profile,
        'persona':     persona,
        'filename':    filename,
        'voice_id':    voice_id,
        'messages':    [],
        'unlocked':    False,
        'unlock_type': None,
    })

def update_session_messages(session_id, messages):
    sb_update('sessions', 'session_id', session_id, {'messages': messages})

def unlock_session(session_id, unlock_type):
    sb_update('sessions', 'session_id', session_id, {
        'unlocked':    True,
        'unlock_type': unlock_type,
    })

# ── File helpers ──────────────────────────────────────────────────────────────

def parse_extension(filename):
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

def allowed_file(filename):
    ext = parse_extension(filename)
    return ext, ext in ALLOWED_EXTENSIONS['video'] or ext in ALLOWED_EXTENSIONS['audio']

def is_video_file(filename):
    return parse_extension(filename) in ALLOWED_EXTENSIONS['video']

def get_mime(filename):
    return MIME_MAP.get(parse_extension(filename), 'application/octet-stream')

# ── Groq helpers ──────────────────────────────────────────────────────────────

def transcribe_groq(file_path, filename):
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(
                'https://api.groq.com/openai/v1/audio/transcriptions',
                headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
                files={'file': (filename, f, get_mime(filename))},
                data={'model': 'whisper-large-v3', 'response_format': 'verbose_json', 'language': 'en'},
                timeout=180,
            )
        data = resp.json()
        if resp.ok:
            return data.get('text', ''), None
        return None, data.get('error', {}).get('message', 'Groq transcription error')
    except Exception as e:
        return None, str(e)

def build_profile_from_transcript(transcript, person_name):
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'
    prompt = f"""You are analyzing a spoken transcript to build a complete memory profile of a person named {person_name}.

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

Extract and write up:
1. VOICE & SPEECH PATTERNS: Speaking pace, accent, regional dialect, favorite phrases, filler words, unique slang, tone, vocabulary style
2. CONVERSATIONS & CONTENT: Full detailed summary of everything discussed, key topics, opinions expressed, jokes, funny moments, memorable lines, specific facts: names, places, references
3. PERSONALITY PROFILE: Energy level and mood, how they talk about others, values, passions, interests, humor style and quirks
4. MEMORABLE DETAILS: Specific stories or anecdotes told, names of friends/family mentioned, strong opinions or recurring beliefs

Return a rich, extremely detailed narrative profile — write as if briefing someone who needs to PERFECTLY IMPERSONATE {person_name} in a conversation. Be very specific. Quote actual phrases and words they used."""
    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.8, 'max_tokens': 2048},
            timeout=60,
        )
        data = resp.json()
        if resp.ok:
            return data['choices'][0]['message']['content'], None
        return None, data.get('error', {}).get('message', 'Groq LLM error')
    except Exception as e:
        return None, str(e)

def chat_as_persona_groq(history, person_name, persona):
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'
    system_prompt = f"""You ARE {person_name}. Not an AI. Not a simulation. You ARE this person, responding from memory.

{persona}

ABSOLUTE RULES:
- Speak ONLY in first person as {person_name}
- Use the exact vocabulary, slang, phrases, and speech patterns described above
- Match their energy level, humor, and personality exactly
- Reference specific memories, people, and events from the analysis naturally
- NEVER say you're an AI, a chatbot, a simulation, or Chronis
- NEVER say "based on the transcript" or "based on the recording"
- If asked something you don't know, respond as {person_name} would — deflect, joke, or say you forgot
- Keep responses conversational, SHORT, natural — NOT essay-like
- Use filler words, informal grammar, the way {person_name} actually talks
- Ask questions back occasionally

You are {person_name}. Start talking."""
    messages = [{'role': 'system', 'content': system_prompt}]
    for msg in history[-20:]:
        messages.append({'role': 'user' if msg['role'] == 'user' else 'assistant', 'content': msg['content']})
    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'llama-3.3-70b-versatile', 'messages': messages, 'temperature': 0.9, 'max_tokens': 512},
            timeout=30,
        )
        data = resp.json()
        if resp.ok:
            return data['choices'][0]['message']['content'], None
        return None, data.get('error', {}).get('message', 'Groq chat error')
    except Exception as e:
        return None, str(e)

# ── Gemini video analysis ─────────────────────────────────────────────────────

def analyze_video_gemini(file_path, filename, person_name):
    if not GEMINI_API_KEY:
        return None, 'GEMINI_API_KEY not set'
    with open(file_path, 'rb') as f:
        b64_data = base64.b64encode(f.read()).decode('utf-8')
    prompt = f"""Analyze this video to build a complete memory profile of a person named {person_name}.
Extract:
1. IDENTITY & APPEARANCE: Physical description, hair, eyes, clothing, accessories, age estimate
2. VOICE & SPEECH: Accent, pace, phrases, filler words, slang, tone, vocabulary style
3. ENVIRONMENT: Setting description, other people present, what they're doing
4. CONVERSATIONS: Full summary of everything discussed, key topics, opinions, jokes, memorable lines
5. PERSONALITY: Energy level, humor style, how they interact, values, interests, quirks
6. MEMORABLE DETAILS: Specific stories told, names mentioned, strong opinions, beliefs
Write a rich, detailed narrative profile for someone who needs to perfectly impersonate {person_name}. Be very specific. Quote actual phrases and words they used."""
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}'
    payload = {
        'contents': [{'parts': [{'inline_data': {'mime_type': get_mime(filename), 'data': b64_data}}, {'text': prompt}]}],
        'generationConfig': {'temperature': 0.85, 'maxOutputTokens': 2048},
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        data = resp.json()
        if resp.ok:
            return data['candidates'][0]['content']['parts'][0]['text'], None
        return None, data.get('error', {}).get('message', 'Gemini error')
    except Exception as e:
        return None, str(e)

# ── XTTS voice cloning (HuggingFace Space) ────────────────────────────────────

def store_voice_reference(audio_path, session_id):
    """
    Store a trimmed audio clip in Supabase Storage as the voice reference.
    Returns (filename, error). Filename used later when calling XTTS.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, 'Supabase not configured'
    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        # Trim to first ~10 seconds (1.6MB at 128kbps)
        audio_bytes = audio_bytes[:1_600_000]
        filename = f'{session_id}_ref.mp3'
        r = requests.post(
            f'{SUPABASE_URL}/storage/v1/object/voice-refs/{filename}',
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'audio/mpeg',
                'x-upsert': 'true',
            },
            data=audio_bytes,
            timeout=30,
        )
        if r.ok:
            return filename, None
        return None, f'Storage error: {r.status_code} {r.text[:200]}'
    except Exception as e:
        return None, str(e)

def get_voice_reference_b64(filename):
    """
    Fetch voice reference bytes from Supabase Storage, return as base64.
    """
    try:
        r = requests.get(
            f'{SUPABASE_URL}/storage/v1/object/voice-refs/{filename}',
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
            },
            timeout=15,
        )
        if r.ok:
            return base64.b64encode(r.content).decode('utf-8'), None
        return None, f'Fetch error: {r.status_code}'
    except Exception as e:
        return None, str(e)

def synthesize_xtts(text, voice_ref_filename):
    if not XTTS_SPACE_URL:
        print('  ❌ XTTS_SPACE_URL not set', flush=True)
        return None, 'XTTS_SPACE_URL not configured'
    ref_b64, err = get_voice_reference_b64(voice_ref_filename)
    if err:
        print(f'  ❌ Could not fetch voice ref: {err}', flush=True)
        return None, f'Could not fetch voice reference: {err}'
    print(f'  → Calling XTTS Space...', flush=True)
    try:
        resp = requests.post(
            f'{XTTS_SPACE_URL}/run/predict',
            json={'data': [text[:500], ref_b64, XTTS_SECRET]},
            timeout=120,
        )
        print(f'  → XTTS response: {resp.status_code}', flush=True)
        if resp.ok:
            result = resp.json()
            data   = result.get('data', [])
            print(f'  → XTTS data length: {len(data)}', flush=True)
            if len(data) >= 2:
                audio_b64 = data[0]
                status    = data[1]
                print(f'  → XTTS status: {status}', flush=True)
                if status == 'ok' and audio_b64:
                    return audio_b64, None
                return None, f'XTTS error: {status}'
        print(f'  ❌ XTTS failed: {resp.status_code} {resp.text[:300]}', flush=True)
        return None, f'XTTS API error: {resp.status_code} {resp.text[:200]}'
    except Exception as e:
        print(f'  ❌ XTTS exception: {str(e)}', flush=True)
        return None, str(e)

# ── Persona builder ───────────────────────────────────────────────────────────

def build_persona(person_name, profile):
    return f"""You ARE {person_name}. Not an AI. Not a simulation. You ARE this person.

Here is everything extracted about who you are:

{profile}

RULES — never break these:
- Speak ONLY in first person as {person_name}
- Use the exact vocabulary, slang, phrases, and speech patterns described above
- Match the energy level, humor, and personality exactly
- Reference specific memories, people, and events from the analysis naturally
- Never say you're an AI, a simulation, or Chronis
- Never say "based on the video/audio/transcript" — you just know these things because they're your memories
- If asked something you don't know, respond as {person_name} would — maybe deflect, joke, or say you forgot
- Keep responses conversational, not essay-like — talk like a real person
- Use filler words, informal grammar, occasional typos — exactly as {person_name} speaks
- Feel free to ask questions back, just as {person_name} would

You are {person_name}. Respond naturally."""

# ── Email helpers ─────────────────────────────────────────────────────────────

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
            json={'from': 'Chronis <hello@chronis.in>', 'to': email, 'subject': f"You're on the list, {first} — Chronis", 'html': html},
            timeout=10)
    except Exception as e:
        print(f'Welcome email error: {e}', flush=True)

def send_notify(person_name, session_id):
    if not RESEND_API_KEY or not NOTIFY_EMAIL:
        return
    try:
        requests.post('https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={'from': 'Chronis Demo <hello@chronis.in>', 'to': NOTIFY_EMAIL,
                  'subject': f'🧠 New demo: {person_name}',
                  'html': f'<p>New session: <b>{person_name}</b> — {session_id[:8]} — {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>'},
            timeout=8)
    except Exception as e:
        print(f'Notify email error: {e}', flush=True)

# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size is 50MB.'}), 413

# ── Static routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/demo')
def demo():
    return app.send_static_file('demo.html')

@app.route('/thankyou')
def thankyou():
    return app.send_static_file('thankyou.html')

@app.route('/admin')
def admin():
    # Protected by login on the frontend (secret checked via API)
    return app.send_static_file('admin.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ts': datetime.utcnow().isoformat()})

# ── Waitlist routes ───────────────────────────────────────────────────────────

@app.route('/api/count')
def api_count():
    return jsonify({'count': get_waitlist_count()})

@app.route('/api/join', methods=['POST'])
def api_join():
    data    = request.get_json(silent=True) or {}
    name    = (data.get('name') or '').strip()
    email   = (data.get('email') or '').strip().lower()
    country = (data.get('country') or '').strip()
    if not name or not email or not country:
        return jsonify({'error': 'All fields are required.'}), 400
    if '@' not in email or '.' not in email.split('@')[-1]:
        return jsonify({'error': 'Please enter a valid email address.'}), 400
    if email_exists(email):
        return jsonify({'error': 'This email is already on the waitlist.'}), 400
    position = get_waitlist_count() + 1
    save_waitlist_entry(name, email, country, position)
    try:
        send_welcome_email(name, email, position)
    except Exception as e:
        print(f'Email send error: {e}', flush=True)
    return jsonify({'success': True, 'count': position}), 200

# ── Stripe payment routes ─────────────────────────────────────────────────────

@app.route('/api/create-checkout', methods=['POST'])
def create_checkout():
    if not STRIPE_SECRET_KEY:
        return jsonify({'error': 'Payments not configured.'}), 500
    data           = request.get_json(silent=True) or {}
    session_id     = data.get('session_id', '')   # only needed for chat unlock
    checkout_type  = data.get('type', 'chat')     # 'chat' or 'video'

    labels = {
        'chat':  ('Continue Conversation', 'Unlimited messages — never lose this connection'),
        'video': ('Video Memory Analysis', 'Full video analysis + unlimited chat'),
    }
    name, desc = labels.get(checkout_type, labels['chat'])

    # For video: no chronis session exists yet, pass empty
    # For chat:  pass real session_id so we can restore conversation after payment
    csession_param = session_id if checkout_type == 'chat' else ''

    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': name, 'description': desc},
                    'unit_amount': 100,  # $1.00
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f'{SITE_URL}/demo?payment=success&csession={csession_param}&type={checkout_type}&ss={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{SITE_URL}/demo?payment=cancelled',
            metadata={'chronis_session': csession_param, 'type': checkout_type},
        )
        return jsonify({'url': checkout.url})
    except Exception as e:
        print(f'Stripe error: {e}', flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-checkout', methods=['POST'])
def verify_checkout():
    if not STRIPE_SECRET_KEY:
        return jsonify({'error': 'Payments not configured.'}), 500
    data              = request.get_json(silent=True) or {}
    stripe_session_id = data.get('stripe_session_id', '')
    chronis_session   = data.get('chronis_session', '')  # empty string for video (no session yet)
    checkout_type     = data.get('type', 'chat')

    try:
        checkout = stripe.checkout.Session.retrieve(stripe_session_id)
        if checkout.payment_status != 'paid':
            return jsonify({'error': 'Payment not completed.'}), 400

        if checkout_type == 'video':
            # VIDEO: no session exists yet — generate a one-time upload token
            # User will include this token when uploading the video file
            video_token = uuid.uuid4().hex
            sb_insert('payment_tokens', {
                'token':              video_token,
                'stripe_session_id':  stripe_session_id,
                'chronis_session_id': None,
                'type':               'video',
                'used':               False,
            })
            print(f'✅ Video payment verified: token {video_token[:8]} issued', flush=True)
            return jsonify({'success': True, 'type': 'video', 'video_token': video_token})

        else:
            # CHAT: real session exists — unlock it directly
            if not chronis_session:
                return jsonify({'error': 'Missing session ID for chat unlock.'}), 400
            unlock_session(chronis_session, checkout_type)
            print(f'✅ Chat payment verified: session {chronis_session[:8]} unlocked', flush=True)
            return jsonify({'success': True, 'type': checkout_type})

    except Exception as e:
        print(f'Stripe verify error: {e}', flush=True)
        return jsonify({'error': 'Verification failed.'}), 400

# ── Session restore route (after Stripe redirect) ─────────────────────────────

@app.route('/api/session/<session_id>', methods=['GET'])
def restore_session(session_id):
    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found or expired.'}), 404
    raw_ts  = session.get('created_at', '')
    created = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
    if datetime.now(timezone.utc) - created > timedelta(hours=6):
        return jsonify({'error': 'Session expired.'}), 400
    messages = session.get('messages') or []
    return jsonify({
        'person_name': session['person_name'],
        'profile':     session['profile'],
        'messages':    messages,
        'unlocked':    session.get('unlocked', False),
        'unlock_type': session.get('unlock_type'),
        'has_voice':   bool(session.get('voice_id')),
    })

# ── Demo: analyze text memory ─────────────────────────────────────────────────

@app.route('/api/analyze-text', methods=['POST'])
def api_analyze_text():
    data        = request.get_json(silent=True) or {}
    person_name = (data.get('person_name') or 'this person').strip()
    memory_text = (data.get('memory_text') or '').strip()

    if not memory_text:
        return jsonify({'error': 'No memory text provided.'}), 400
    if len(memory_text) < 30:
        return jsonify({'error': 'Please write at least a few sentences.'}), 400
    if len(memory_text) > 10000:
        return jsonify({'error': 'Text too long. Max 10,000 characters.'}), 400

    print(f'📝 Analyzing text memory for "{person_name}" ({len(memory_text)} chars)', flush=True)

    # Build profile directly from written memory using Groq LLaMA
    # No transcription needed — treat the text as the source material
    if not GROQ_API_KEY:
        return jsonify({'error': 'GROQ_API_KEY not set'}), 500

    prompt = f"""You are building a memory profile of a person named {person_name} based on a written description provided by someone who knows them.

WRITTEN MEMORY:
\"\"\"
{memory_text}
\"\"\"

Based on this description, extract and write up:

1. VOICE & SPEECH PATTERNS:
   - How they likely spoke — inferred from the description
   - Any phrases, words, or expressions mentioned
   - Their tone: warm, funny, serious, sarcastic, etc.

2. PERSONALITY PROFILE:
   - Their energy level and mood as described
   - How they treated people, what they valued
   - Their sense of humor, quirks, habits
   - What made them uniquely them

3. MEMORIES & STORIES:
   - Specific stories, moments, or events described
   - Names of people, places mentioned
   - Opinions, beliefs, things they loved or hated

4. HOW THEY WOULD TALK:
   - Based on everything above, how would {person_name} naturally respond in conversation?
   - What topics would they bring up?
   - How would they greet someone? How would they joke?

Return a rich, detailed narrative profile — write as if briefing someone who needs to PERFECTLY IMPERSONATE {person_name} in a heartfelt conversation. Make it feel real and human. Be specific — infer personality from every detail given."""

    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'llama-3.3-70b-versatile',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.85,
                'max_tokens': 2048,
            },
            timeout=60,
        )
        resp_data = resp.json()
        if not resp.ok:
            return jsonify({'error': resp_data.get('error', {}).get('message', 'Groq error')}), 500
        profile = resp_data['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    session_id = str(uuid.uuid4())
    persona    = build_persona(person_name, profile)
    create_session(session_id, person_name, profile, persona, 'text_memory', None)

    try:
        send_notify(person_name, session_id)
    except Exception:
        pass

    return jsonify({
        'session_id':  session_id,
        'person_name': person_name,
        'profile':     profile,
        'has_voice':   False,
    })

# ── Demo: analyze ─────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file        = request.files['file']
    person_name = (request.form.get('person_name') or 'this person').strip()

    if not file.filename or len(file.filename) > 200:
        return jsonify({'error': 'Invalid filename.'}), 400

    ext, valid = allowed_file(file.filename)
    if not valid:
        return jsonify({'error': 'Unsupported file type. Use MP4, MOV, MP3, WAV, M4A, etc.'}), 400

    # Check admin bypass
    admin_secret_form = (request.form.get('admin_secret') or '').strip()
    is_admin = (admin_secret_form == ADMIN_SECRET and bool(ADMIN_SECRET))

    # Video requires a paid unlock token (unless admin)
    if is_video_file(file.filename) and not is_admin:
        unlock_token = (request.form.get('unlock_token') or '').strip()
        if not unlock_token:
            return jsonify({'error': 'payment_required', 'message': 'Video analysis requires payment.'}), 402
        rows = sb_select('payment_tokens', f'token=eq.{unlock_token}&used=eq.false&select=id')
        if not rows:
            return jsonify({'error': 'Invalid or already used payment token.'}), 402
        sb_update('payment_tokens', 'token', unlock_token, {'used': True})

    session_id = str(uuid.uuid4())
    safe_name  = f'{session_id}.{ext}'
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(file_path)

    print(f'📁 Analyzing {file.filename} as "{person_name}" ({session_id[:8]})', flush=True)

    try:
        if is_video_file(file.filename):
            print('  → Sending video to Gemini 1.5 Flash', flush=True)
            profile, error = analyze_video_gemini(file_path, safe_name, person_name)
            voice_id = None
        else:
            print('  → Transcribing audio with Groq Whisper', flush=True)
            transcript, error = transcribe_groq(file_path, safe_name)
            if error:
                return jsonify({'error': f'Transcription failed: {error}'}), 500
            print(f'  → Transcript: {len(transcript)} chars. Building profile...', flush=True)
            profile, error = build_profile_from_transcript(transcript, person_name)
            voice_id = None
            if not error and XTTS_SPACE_URL:
                print('  → Storing voice reference for XTTS', flush=True)
                voice_id, v_err = store_voice_reference(file_path, session_id)
                if v_err:
                    print(f'  ⚠️  Voice ref storage failed: {v_err}', flush=True)
                else:
                    print(f'  ✅ Voice reference stored: {voice_id}', flush=True)

        if error:
            return jsonify({'error': f'Analysis failed: {error}'}), 500
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    persona = build_persona(person_name, profile)
    create_session(session_id, person_name, profile, persona, file.filename, voice_id)

    try:
        send_notify(person_name, session_id)
    except Exception:
        pass

    return jsonify({
        'session_id':  session_id,
        'person_name': person_name,
        'profile':     profile,
        'has_voice':   voice_id is not None,
    })

# ── Demo: chat ────────────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    user_msg   = (data.get('message') or '').strip()

    if not session_id or not user_msg:
        return jsonify({'error': 'Missing session_id or message.'}), 400
    if len(user_msg) > 500:
        return jsonify({'error': 'Message too long (max 500 chars).'}), 400

    # Admin bypass — skip all limits
    is_admin = (data.get('admin_secret', '') == ADMIN_SECRET and bool(ADMIN_SECRET))

    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found. Please upload again.'}), 404

    raw_ts  = session.get('created_at', '')
    created = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
    if not is_admin and datetime.now(timezone.utc) - created > timedelta(hours=6):
        return jsonify({'error': 'Session expired. Please upload again.'}), 400

    history     = session.get('messages') or []
    unlocked    = session.get('unlocked', False) or is_admin
    person_name = session['person_name']
    persona     = session['persona']
    voice_id    = session.get('voice_id')

    # Count actual user exchanges (not total messages)
    exchanges = len([m for m in history if m['role'] == 'user'])

    # Enforce free limit for locked sessions (admin bypasses)
    if not unlocked and exchanges >= FREE_MSG_LIMIT:
        return jsonify({
            'error':          'limit_reached',
            'message':        f'Free limit of {FREE_MSG_LIMIT} messages reached.',
            'session_id':     session_id,
            'person_name':    person_name,
        }), 402

    if len(history) >= 100:
        return jsonify({'error': 'Session message limit reached. Please start a new session.'}), 400

    history.append({'role': 'user', 'content': user_msg, 'ts': datetime.utcnow().isoformat()})
    reply, error = chat_as_persona_groq(history, person_name, persona)
    if error:
        return jsonify({'error': f'AI error: {error}'}), 500

    history.append({'role': 'assistant', 'content': reply, 'ts': datetime.utcnow().isoformat()})
    update_session_messages(session_id, history)

    exchanges_after = exchanges + 1

    return jsonify({
        'reply':           reply,
        'person_name':     person_name,
        'has_voice':       voice_id is not None,
        'messages_used':   exchanges_after,
        'free_limit':      FREE_MSG_LIMIT,
        'limit_reached':   (not unlocked) and (exchanges_after >= FREE_MSG_LIMIT),
        'unlocked':        unlocked,
    })

# ── Voice synthesis ───────────────────────────────────────────────────────────

@app.route('/api/speak', methods=['POST'])
def api_speak():
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    text       = (data.get('text') or '').strip()
    if not session_id or not text:
        return jsonify({'error': 'Missing session_id or text.'}), 400
    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found.'}), 404
    voice_id = session.get('voice_id')
    if not voice_id:
        return jsonify({'error': 'No cloned voice for this session.'}), 400
    audio_b64, error = synthesize_xtts(text, voice_id)
    if error:
        return jsonify({'error': error}), 500
    return jsonify({'audio': audio_b64, 'format': 'wav'})

# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route('/api/sessions')
def api_sessions():
    secret = request.args.get('secret', '')
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    rows = sb_select('sessions', 'select=session_id,person_name,created_at,filename,voice_id,messages,unlocked,unlock_type&order=created_at.desc&limit=200')
    summary = []
    for s in rows:
        msgs = s.get('messages') or []
        summary.append({
            'session_id':  s['session_id'][:8],
            'person_name': s.get('person_name'),
            'created_at':  s.get('created_at'),
            'messages':    len(msgs),
            'filename':    s.get('filename'),
            'has_voice':   bool(s.get('voice_id')),
            'unlocked':    s.get('unlocked', False),
            'unlock_type': s.get('unlock_type'),
        })
    return jsonify({'sessions': summary, 'total': len(summary)})

@app.route('/api/waitlist')
def api_waitlist():
    secret = request.args.get('secret', '')
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    rows = sb_select('waitlist', 'select=*&order=created_at.asc')
    return jsonify({'entries': rows, 'total': len(rows)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Flask app starting on port {port}', flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)