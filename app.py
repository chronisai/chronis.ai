import os
import sys
import json
import uuid
import base64
import requests
import tempfile
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
SUPABASE_URL       = os.environ.get('SUPABASE_URL', '')       # e.g. https://xxxx.supabase.co
SUPABASE_KEY       = os.environ.get('SUPABASE_SERVICE_KEY', '') # service_role key (not anon)

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

# ── Supabase REST helpers ─────────────────────────────────────────────────────

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
    """SELECT rows. Returns list or []."""
    try:
        url = f'{SUPABASE_URL}/rest/v1/{table}?{query}'
        r = requests.get(url, headers=_sb_headers(), timeout=10)
        return r.json() if r.ok else []
    except Exception as e:
        print(f'[Supabase select error] {e}', flush=True)
        return []

def sb_count(table, query=''):
    """Return exact row count for a table."""
    try:
        url = f'{SUPABASE_URL}/rest/v1/{table}?select=id&{query}'
        r = requests.get(url, headers=_sb_headers({'Prefer': 'count=exact'}), timeout=10)
        cr = r.headers.get('Content-Range', '0/0')
        return int(cr.split('/')[-1])
    except Exception as e:
        print(f'[Supabase count error] {e}', flush=True)
        return 0

def sb_insert(table, data):
    """INSERT a row. Returns inserted row or None."""
    try:
        r = requests.post(
            f'{SUPABASE_URL}/rest/v1/{table}',
            headers=_sb_headers(),
            json=data,
            timeout=10
        )
        rows = r.json()
        return rows[0] if r.ok and rows else None
    except Exception as e:
        print(f'[Supabase insert error] {e}', flush=True)
        return None

def sb_update(table, match_col, match_val, data):
    """UPDATE rows matching match_col=match_val."""
    try:
        r = requests.patch(
            f'{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}',
            headers=_sb_headers(),
            json=data,
            timeout=10
        )
        return r.ok
    except Exception as e:
        print(f'[Supabase update error] {e}', flush=True)
        return False

# ── Waitlist helpers ──────────────────────────────────────────────────────────

WAITLIST_BASELINE = 93  # existing count before Supabase migration

def get_waitlist_count():
    count = sb_count('waitlist')
    return max(count + WAITLIST_BASELINE, WAITLIST_BASELINE)

def email_exists(email):
    rows = sb_select('waitlist', f'email=eq.{requests.utils.quote(email)}&select=id')
    return len(rows) > 0

def save_waitlist_entry(name, email, country, position):
    sb_insert('waitlist', {
        'name': name,
        'email': email,
        'country': country,
        'position': position,
    })

# ── Session helpers ───────────────────────────────────────────────────────────

def get_session(session_id):
    rows = sb_select('sessions', f'session_id=eq.{session_id}&select=*')
    return rows[0] if rows else None

def create_session(session_id, person_name, profile, persona, filename, voice_id=None):
    sb_insert('sessions', {
        'session_id': session_id,
        'person_name': person_name,
        'profile': profile,
        'persona': persona,
        'filename': filename,
        'voice_id': voice_id,
        'messages': [],
    })

def update_session_messages(session_id, messages):
    sb_update('sessions', 'session_id', session_id, {'messages': messages})

# ── File helpers ──────────────────────────────────────────────────────────────

def parse_extension(filename):
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

def allowed_file(filename):
    ext = parse_extension(filename)
    valid = ext in ALLOWED_EXTENSIONS['video'] or ext in ALLOWED_EXTENSIONS['audio']
    return ext, valid

def is_video_file(filename):
    return parse_extension(filename) in ALLOWED_EXTENSIONS['video']

def get_mime(filename):
    return MIME_MAP.get(parse_extension(filename), 'application/octet-stream')

# ── Groq: audio transcription ─────────────────────────────────────────────────

def transcribe_groq(file_path, filename):
    """
    Transcribe an audio file using Groq Whisper-large-v3.
    Free tier: 2 hours/day audio. Returns (transcript_str, error).
    """
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'
    mime = get_mime(filename)
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(
                'https://api.groq.com/openai/v1/audio/transcriptions',
                headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
                files={'file': (filename, f, mime)},
                data={
                    'model': 'whisper-large-v3',
                    'response_format': 'verbose_json',
                    'language': 'en',
                },
                timeout=180,
            )
        data = resp.json()
        if resp.ok:
            return data.get('text', ''), None
        return None, data.get('error', {}).get('message', 'Groq transcription error')
    except Exception as e:
        return None, str(e)

# ── Groq: build persona from transcript ──────────────────────────────────────

def build_profile_from_transcript(transcript, person_name):
    """
    Use Groq LLaMA-3.3-70b to build a rich persona profile from a transcript.
    Returns (profile_str, error).
    """
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'

    prompt = f"""You are analyzing a spoken transcript to build a complete memory profile of a person named {person_name}.

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

Extract and write up:

1. VOICE & SPEECH PATTERNS:
   - Speaking pace, accent, regional dialect
   - Favorite phrases, filler words, unique slang
   - Tone: serious/jokey/warm/sarcastic/etc.
   - Vocabulary level and style

2. CONVERSATIONS & CONTENT:
   - Full detailed summary of everything discussed
   - Key topics, opinions expressed
   - Jokes, funny moments, memorable lines
   - Specific facts: names, places, sports teams, references

3. PERSONALITY PROFILE:
   - Energy level and mood
   - How they talk about others
   - Values, passions, interests
   - Humor style and quirks

4. MEMORABLE DETAILS:
   - Specific stories or anecdotes told
   - Names of friends/family mentioned
   - Strong opinions or recurring beliefs

Return a rich, extremely detailed narrative profile — write as if briefing someone who needs to PERFECTLY IMPERSONATE {person_name} in a conversation. Be very specific. Quote actual phrases and words they used."""

    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'llama-3.3-70b-versatile',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.8,
                'max_tokens': 2048,
            },
            timeout=60,
        )
        data = resp.json()
        if resp.ok:
            return data['choices'][0]['message']['content'], None
        return None, data.get('error', {}).get('message', 'Groq LLM error')
    except Exception as e:
        return None, str(e)

# ── Groq: persona chat ────────────────────────────────────────────────────────

def chat_as_persona_groq(history, person_name, persona):
    """
    Chat using Groq LLaMA with a full persona system prompt.
    history: list of {role, content} dicts (user/assistant).
    Returns (reply_str, error).
    """
    if not GROQ_API_KEY:
        return None, 'GROQ_API_KEY not set'

    system_prompt = f"""You ARE {person_name}. Not an AI. Not a simulation. You ARE this person, responding from memory.

{persona}

ABSOLUTE RULES — never break these:
- Speak ONLY in first person as {person_name}
- Use the exact vocabulary, slang, phrases, and speech patterns described above
- Match their energy level, humor, and personality exactly
- Reference specific memories, people, and events from the analysis naturally
- NEVER say you're an AI, a chatbot, a simulation, or Chronis
- NEVER say "based on the transcript" or "based on the recording" — you just know these things
- If asked something you don't know, respond as {person_name} would — deflect, joke, or say you forgot
- Keep responses conversational, SHORT, natural — NOT essay-like
- Use filler words, informal grammar, the way {person_name} actually talks
- Ask questions back occasionally, just as {person_name} would

You are {person_name}. Start talking."""

    messages = [{'role': 'system', 'content': system_prompt}]
    for msg in history[-20:]:
        messages.append({
            'role': 'user' if msg['role'] == 'user' else 'assistant',
            'content': msg['content'],
        })

    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'llama-3.3-70b-versatile',
                'messages': messages,
                'temperature': 0.9,
                'max_tokens': 512,
            },
            timeout=30,
        )
        data = resp.json()
        if resp.ok:
            return data['choices'][0]['message']['content'], None
        return None, data.get('error', {}).get('message', 'Groq chat error')
    except Exception as e:
        return None, str(e)

# ── Gemini: video analysis (video only) ──────────────────────────────────────

def analyze_video_gemini(file_path, filename, person_name):
    """
    Analyze a video file using Gemini 1.5 Flash (multimodal, handles video natively).
    Returns (profile_str, error).
    """
    if not GEMINI_API_KEY:
        return None, 'GEMINI_API_KEY not set'

    mime = get_mime(filename)
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
        'contents': [{
            'parts': [
                {'inline_data': {'mime_type': mime, 'data': b64_data}},
                {'text': prompt},
            ]
        }],
        'generationConfig': {'temperature': 0.85, 'maxOutputTokens': 2048},
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        data = resp.json()
        if resp.ok:
            return data['candidates'][0]['content']['parts'][0]['text'], None
        err = data.get('error', {}).get('message', 'Gemini error')
        return None, err
    except Exception as e:
        return None, str(e)

# ── ElevenLabs: voice cloning ─────────────────────────────────────────────────

def clone_voice_elevenlabs(audio_path, person_name):
    """
    Clone a voice using ElevenLabs Instant Voice Cloning (free tier: 1 custom voice).
    Returns (voice_id, error).
    """
    if not ELEVENLABS_API_KEY:
        return None, 'ELEVENLABS_API_KEY not set'
    try:
        with open(audio_path, 'rb') as f:
            resp = requests.post(
                'https://api.elevenlabs.io/v1/voices/add',
                headers={'xi-api-key': ELEVENLABS_API_KEY},
                data={
                    'name': f'{person_name}_{uuid.uuid4().hex[:6]}',
                    'description': f'Cloned voice of {person_name} — Chronis',
                },
                files={'files': (os.path.basename(audio_path), f, get_mime(audio_path))},
                timeout=90,
            )
        data = resp.json()
        if resp.ok:
            return data.get('voice_id'), None
        detail = data.get('detail', {})
        msg = detail.get('message', str(detail)) if isinstance(detail, dict) else str(detail)
        return None, msg
    except Exception as e:
        return None, str(e)

def synthesize_speech_elevenlabs(text, voice_id):
    """
    Convert text to speech using the cloned ElevenLabs voice.
    Returns (audio_base64, error).
    """
    if not ELEVENLABS_API_KEY:
        return None, 'ELEVENLABS_API_KEY not set'
    try:
        resp = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers={
                'xi-api-key': ELEVENLABS_API_KEY,
                'Content-Type': 'application/json',
            },
            json={
                'text': text[:1000],
                'model_id': 'eleven_turbo_v2',
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.75,
                    'style': 0.0,
                    'use_speaker_boost': True,
                },
            },
            timeout=30,
        )
        if resp.ok:
            return base64.b64encode(resp.content).decode('utf-8'), None
        return None, f'ElevenLabs TTS error: {resp.status_code}'
    except Exception as e:
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

# ── Email ─────────────────────────────────────────────────────────────────────

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
        requests.post(
            'https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'from': 'Chronis <hello@chronis.in>',
                'to': email,
                'subject': f"You're on the list, {first} — Chronis",
                'html': html,
            },
            timeout=10,
        )
    except Exception as e:
        print(f'Welcome email error: {e}', flush=True)

def send_notify(person_name, session_id):
    if not RESEND_API_KEY or not NOTIFY_EMAIL:
        return
    html = f"""
<div style="font-family:monospace;background:#0a0a0a;color:#fff;padding:32px;border-radius:12px;max-width:480px;">
  <p style="color:rgba(255,255,255,0.4);font-size:11px;letter-spacing:3px;text-transform:uppercase;margin:0 0 16px;">New Chronis Demo Session</p>
  <table style="width:100%;border-collapse:collapse;">
    <tr><td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:rgba(255,255,255,0.4);font-size:13px;width:120px;">Person</td>
        <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:#fff;font-size:13px;">{person_name}</td></tr>
    <tr><td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:rgba(255,255,255,0.4);font-size:13px;">Session</td>
        <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);color:#fff;font-size:13px;">{session_id[:8]}</td></tr>
    <tr><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:13px;">Time</td>
        <td style="padding:8px 0;color:#fff;font-size:13px;">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</td></tr>
  </table>
</div>"""
    try:
        requests.post(
            'https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'from': 'Chronis Demo <hello@chronis.in>',
                'to': NOTIFY_EMAIL,
                'subject': f'🧠 New demo: {person_name}',
                'html': html,
            },
            timeout=8,
        )
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

# ── Demo: analyze route ───────────────────────────────────────────────────────

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

    session_id = str(uuid.uuid4())
    safe_name  = f'{session_id}.{ext}'
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(file_path)

    print(f'📁 Analyzing {file.filename} as "{person_name}" ({session_id[:8]})', flush=True)

    try:
        if is_video_file(file.filename):
            # ── VIDEO: Gemini handles visual + audio together ──────────────
            print('  → Sending video to Gemini 1.5 Flash', flush=True)
            profile, error = analyze_video_gemini(file_path, safe_name, person_name)
            voice_id = None  # Voice cloning not available for video uploads
        else:
            # ── AUDIO: Groq Whisper transcription → Groq LLaMA profile ────
            print('  → Transcribing audio with Groq Whisper', flush=True)
            transcript, error = transcribe_groq(file_path, safe_name)
            if error:
                return jsonify({'error': f'Transcription failed: {error}'}), 500

            print(f'  → Transcript: {len(transcript)} chars. Building profile...', flush=True)
            profile, error = build_profile_from_transcript(transcript, person_name)

            # ── Voice cloning (audio only) ──────────────────────────────────
            voice_id = None
            if not error and ELEVENLABS_API_KEY:
                print('  → Cloning voice with ElevenLabs', flush=True)
                voice_id, v_err = clone_voice_elevenlabs(file_path, person_name)
                if v_err:
                    print(f'  ⚠️  Voice clone failed: {v_err}', flush=True)
                else:
                    print(f'  ✅ Voice cloned: {voice_id}', flush=True)

        if error:
            return jsonify({'error': f'Analysis failed: {error}'}), 500

    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    persona = build_persona(person_name, profile)

    # Persist session in Supabase
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

# ── Demo: chat route ──────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    user_msg   = (data.get('message') or '').strip()

    if not session_id or not user_msg:
        return jsonify({'error': 'Missing session_id or message.'}), 400
    if len(user_msg) > 500:
        return jsonify({'error': 'Message too long (max 500 chars).'}), 400

    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found. Please upload again.'}), 404

    # Expiry check (6 hours)
    raw_ts  = session.get('created_at', '')
    created = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
    if datetime.now(timezone.utc) - created > timedelta(hours=6):
        return jsonify({'error': 'Session expired. Please upload again.'}), 400

    history = session.get('messages') or []
    if len(history) >= 50:
        return jsonify({'error': 'Session message limit reached. Please start a new session.'}), 400

    person_name = session['person_name']
    persona     = session['persona']
    voice_id    = session.get('voice_id')

    # Append user message, get reply
    history.append({'role': 'user', 'content': user_msg, 'ts': datetime.utcnow().isoformat()})
    reply, error = chat_as_persona_groq(history, person_name, persona)
    if error:
        return jsonify({'error': f'AI error: {error}'}), 500

    history.append({'role': 'assistant', 'content': reply, 'ts': datetime.utcnow().isoformat()})
    update_session_messages(session_id, history)

    return jsonify({
        'reply':       reply,
        'person_name': person_name,
        'has_voice':   voice_id is not None,
    })

# ── Voice synthesis route ─────────────────────────────────────────────────────

@app.route('/api/speak', methods=['POST'])
def api_speak():
    """
    Convert a chat reply to speech using the session's cloned voice.
    Body: { session_id, text }
    Response: { audio: <base64 mp3>, format: 'mp3' }
    """
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
        return jsonify({'error': 'No cloned voice for this session. Upload an audio file to enable voice.'}), 400

    audio_b64, error = synthesize_speech_elevenlabs(text, voice_id)
    if error:
        return jsonify({'error': error}), 500

    return jsonify({'audio': audio_b64, 'format': 'mp3'})

# ── Admin: sessions overview ──────────────────────────────────────────────────

@app.route('/api/sessions')
def api_sessions():
    secret = request.args.get('secret', '')
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401

    rows = sb_select('sessions', 'select=session_id,person_name,created_at,filename,voice_id,messages&order=created_at.desc&limit=200')
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
        })
    return jsonify({'sessions': summary, 'total': len(summary)})

# ── Admin: waitlist export ────────────────────────────────────────────────────

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