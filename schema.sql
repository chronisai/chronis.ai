-- ============================================================
-- Chronis V2 — Supabase Schema
-- ============================================================
-- Run this in the Supabase SQL editor (Dashboard → SQL Editor).
-- Safe to re-run: all statements use IF NOT EXISTS / OR REPLACE.
--
-- Table creation order matters — foreign keys reference earlier tables.
-- 1. users       (references Supabase auth.users)
-- 2. agents      (references users)
-- 3. voices      (references agents)
-- 4. sessions    (references users + agents)
-- 5. memories    (references agents)
-- 6. session_events (references sessions)
-- ============================================================


-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()


-- ── Enums ─────────────────────────────────────────────────────────────────────

DO $$ BEGIN
  CREATE TYPE agent_status AS ENUM ('creating', 'ready', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE voice_status AS ENUM ('processing', 'ready', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE session_state AS ENUM (
    'idle', 'listening', 'thinking', 'speaking', 'interrupted', 'ending', 'ended'
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE memory_type AS ENUM ('identity', 'pinned', 'summary', 'conversation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ── 1. users ──────────────────────────────────────────────────────────────────
-- Thin wrapper around Supabase auth.users.
-- Created automatically on first sign-in via the trigger below.

CREATE TABLE IF NOT EXISTS public.users (
  id           uuid        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email        text        NOT NULL UNIQUE,
  created_at   timestamptz NOT NULL DEFAULT now()
);

-- Auto-create a public.users row when a new auth user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.users (id, email)
  VALUES (new.id, new.email)
  ON CONFLICT (id) DO NOTHING;
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ── 2. agents ─────────────────────────────────────────────────────────────────
-- One row per digital replica. A user can have multiple agents.

CREATE TABLE IF NOT EXISTS public.agents (
  id              uuid          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid          NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  simli_agent_id  text,                         -- Simli's face model ID. NULL until created.
  name            text          NOT NULL,        -- Human name of the replica ("Mom", "Dad")
  personality     text,                          -- Full system prompt / personality description
  status          agent_status  NOT NULL DEFAULT 'creating',
  photo_url       text,                          -- Supabase storage path to validated photo
  created_at      timestamptz   NOT NULL DEFAULT now(),
  updated_at      timestamptz   NOT NULL DEFAULT now()
);

-- Index on user_id — every agent query filters by this
CREATE INDEX IF NOT EXISTS agents_user_id_idx ON public.agents(user_id);
-- Index on status — stale job recovery queries this
CREATE INDEX IF NOT EXISTS agents_status_idx  ON public.agents(status);

-- Auto-update updated_at on any change
CREATE OR REPLACE FUNCTION public.touch_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS agents_touch_updated_at ON public.agents;
CREATE TRIGGER agents_touch_updated_at
  BEFORE UPDATE ON public.agents
  FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();


-- ── 3. voices ─────────────────────────────────────────────────────────────────
-- Voice reference for XTTS v2. One voice per agent.
-- Stores the Modal volume path + quality metrics from validation.

CREATE TABLE IF NOT EXISTS public.voices (
  id               uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id         uuid         NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
  modal_voice_ref  text,                        -- Path on Modal persistent volume
  duration_seconds float,                        -- Validated audio duration (logged for debug)
  snr_db           float,                        -- Signal-to-noise ratio at upload time
  speech_ratio     float,                        -- Fraction of audio that is actual speech
  status           voice_status NOT NULL DEFAULT 'processing',
  created_at       timestamptz  NOT NULL DEFAULT now()
);

-- Index on agent_id — always joined from agents table
CREATE INDEX IF NOT EXISTS voices_agent_id_idx ON public.voices(agent_id);


-- ── 4. sessions ───────────────────────────────────────────────────────────────
-- One row per live session. Created at session start, ended_at written on cleanup.
-- state mirrors SessionController.state in the Python pipeline.

CREATE TABLE IF NOT EXISTS public.sessions (
  id              uuid          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid          REFERENCES public.users(id) ON DELETE SET NULL,
  agent_id        uuid          NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
  daily_room_url  text,                          -- Daily.co room URL (created at session start)
  state           session_state NOT NULL DEFAULT 'idle',
  turn_count      int           NOT NULL DEFAULT 0,
  started_at      timestamptz   NOT NULL DEFAULT now(),
  ended_at        timestamptz,                   -- NULL while active; written on cleanup

  -- V1 demo columns (preserved for backward compatibility)
  session_id      text          UNIQUE,          -- Legacy string session ID from V1
  person_name     text,
  profile         text,
  persona         text,
  filename        text,
  voice_id        text,
  messages        jsonb         DEFAULT '[]',
  unlocked        boolean       DEFAULT false,
  unlock_type     text,
  created_at      timestamptz   NOT NULL DEFAULT now()
);

-- Indexes — these queries run constantly in production
CREATE INDEX IF NOT EXISTS sessions_user_id_idx   ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_agent_id_idx  ON public.sessions(agent_id);
CREATE INDEX IF NOT EXISTS sessions_state_idx     ON public.sessions(state);
-- V1 compatibility index
CREATE INDEX IF NOT EXISTS sessions_session_id_idx ON public.sessions(session_id);


-- ── 5. memories ───────────────────────────────────────────────────────────────
-- Persistent memory for each agent across all sessions.
-- Four types form the context window structure:
--   identity     — who this replica is. Always at top. Never removed.
--   pinned       — key facts. Always present. Never removed.
--   summary      — auto-generated every 20 turns. Replaces raw conversation entries.
--   conversation — raw per-turn exchanges. Slides off after summarization.

CREATE TABLE IF NOT EXISTS public.memories (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id    uuid        NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
  type        memory_type NOT NULL,
  content     text        NOT NULL,
  turn_index  int         NOT NULL DEFAULT 0,   -- Ordering for sliding window
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- Index on agent_id — every context build queries this
CREATE INDEX IF NOT EXISTS memories_agent_id_idx ON public.memories(agent_id);
-- Compound index for type-filtered queries (e.g. fetch all 'conversation' for agent)
CREATE INDEX IF NOT EXISTS memories_agent_type_idx ON public.memories(agent_id, type);
-- Index on turn_index for ordered queries and sliding window cleanup
CREATE INDEX IF NOT EXISTS memories_turn_index_idx ON public.memories(agent_id, turn_index);


-- ── 6. session_events ─────────────────────────────────────────────────────────
-- Flight recorder for the real-time pipeline.
-- REQUIRED — not optional. Without this table, production bugs are impossible
-- to debug. Every state transition, interrupt, and pipeline event lands here.
--
-- Written by the background log worker in main_v2.py.
-- Never written inline from hot pipeline paths (would add DB latency to TTS).
--
-- timestamptz(3) = millisecond precision. Use this for ordering events
-- within the same 20ms VAD frame window.

CREATE TABLE IF NOT EXISTS public.session_events (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id  uuid        REFERENCES public.sessions(id) ON DELETE CASCADE,
  event_type  text        NOT NULL,   -- e.g. 'state.listening_to_thinking'
  payload     jsonb       DEFAULT '{}',
  pipeline    text        DEFAULT '', -- which pipeline emitted: audio|llm|tts|avatar|controller
  created_at  timestamptz(3) NOT NULL DEFAULT now()  -- millisecond precision
);

-- Index on session_id — you will filter by this constantly when debugging
CREATE INDEX IF NOT EXISTS session_events_session_id_idx ON public.session_events(session_id);
-- Index on event_type for cross-session analysis (e.g. how often do interrupts fire?)
CREATE INDEX IF NOT EXISTS session_events_type_idx ON public.session_events(event_type);
-- Index on created_at for time-ordered log queries
CREATE INDEX IF NOT EXISTS session_events_created_at_idx ON public.session_events(created_at DESC);


-- ── 7. waitlist (V1 — preserved) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.waitlist (
  id         uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  name       text        NOT NULL,
  email      text        NOT NULL UNIQUE,
  country    text,
  position   int,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS waitlist_email_idx ON public.waitlist(email);


-- ── 8. payment_tokens (V1 — preserved) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.payment_tokens (
  id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  token               text        NOT NULL UNIQUE,
  stripe_session_id   text,
  chronis_session_id  text,
  type                text        NOT NULL DEFAULT 'chat',
  used                boolean     NOT NULL DEFAULT false,
  created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS payment_tokens_token_idx ON public.payment_tokens(token);


-- ── Row Level Security ─────────────────────────────────────────────────────────
-- Enable RLS on all user-data tables.
-- The backend uses the service role key (bypasses RLS for all operations).
-- If you ever expose Supabase directly to the frontend, add user-facing policies.

ALTER TABLE public.users         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agents        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.voices        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.memories      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_events ENABLE ROW LEVEL SECURITY;

-- Service role bypass (already the default, but explicit is better)
-- Frontend uses anon key → blocked by RLS
-- Backend uses service key → bypasses RLS automatically


-- ── Stale job recovery helper view ────────────────────────────────────────────
-- Run this query to find stuck onboarding jobs:
--   SELECT * FROM public.stale_onboarding_jobs;

CREATE OR REPLACE VIEW public.stale_onboarding_jobs AS
SELECT
  'agent' AS table_name,
  id,
  user_id,
  name,
  status::text,
  created_at,
  now() - created_at AS age
FROM public.agents
WHERE status = 'creating'
  AND created_at < now() - INTERVAL '10 minutes'
UNION ALL
SELECT
  'voice' AS table_name,
  v.id,
  a.user_id,
  a.name,
  v.status::text,
  v.created_at,
  now() - v.created_at AS age
FROM public.voices v
JOIN public.agents a ON a.id = v.agent_id
WHERE v.status = 'processing'
  AND v.created_at < now() - INTERVAL '10 minutes';


-- ── Seed data for local dev ────────────────────────────────────────────────────
-- Uncomment to insert a test user + agent for smoke testing.
-- Replace UUIDs with real auth.users IDs from your Supabase dashboard.
--
-- INSERT INTO public.users (id, email) VALUES
--   ('00000000-0000-0000-0000-000000000001', 'test@chronis.in')
--   ON CONFLICT DO NOTHING;
--
-- INSERT INTO public.agents (id, user_id, name, personality, status) VALUES
--   ('00000000-0000-0000-0000-000000000002',
--    '00000000-0000-0000-0000-000000000001',
--    'Test Agent',
--    'You are a test agent for development purposes.',
--    'ready')
--   ON CONFLICT DO NOTHING;


-- ============================================================
-- Schema complete.
-- Verify with:
--   SELECT table_name FROM information_schema.tables
--   WHERE table_schema = 'public' ORDER BY table_name;
-- ============================================================


-- ── Supabase Storage Buckets ───────────────────────────────────────────────────
-- Run these in the Supabase SQL editor AFTER the tables above.
-- Creates the two storage buckets needed by the onboarding pipeline.

-- agent-photos: stores validated face photos for Simli agent creation
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'agent-photos',
  'agent-photos',
  false,                          -- private: only service role can read
  20971520,                       -- 20MB limit per file
  ARRAY['image/jpeg', 'image/png', 'image/webp']
)
ON CONFLICT (id) DO NOTHING;

-- voice-refs: stores voice reference audio for V1 XTTS HuggingFace Space
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'voice-refs',
  'voice-refs',
  false,
  104857600,                      -- 100MB limit
  ARRAY['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/flac']
)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: only service role can read/write (backend uses service key)
-- No policies needed for anon access — all storage ops go through the backend.
