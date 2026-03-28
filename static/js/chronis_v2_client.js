/**
 * static/js/chronis_v2_client.js
 *
 * Frontend client for the Chronis V2 live session pipeline.
 *
 * Responsibilities:
 *   1. Call POST /api/v2/session/start to get room_url + session_id
 *   2. Join the Daily.co room (audio capture with EC/NS/AGC)
 *   3. Open WebSocket to /ws/session/{session_id}
 *   4. Stream raw PCM audio from Daily.co to the backend WebSocket
 *   5. Send heartbeat every 30 seconds
 *   6. Receive state updates and partial transcripts
 *   7. Clean up on end
 *
 * Audio routing:
 *   Daily.co provides EC + NS + AGC for free in the room config.
 *   We capture the local mic via Daily's MediaStreamTrack API,
 *   pipe it through a ScriptProcessorNode (or AudioWorklet) to
 *   extract raw PCM, and send it to the backend WebSocket as
 *   binary frames.
 *
 * Usage:
 *   import { ChronisV2Client } from './chronis_v2_client.js';
 *
 *   const client = new ChronisV2Client({
 *     agentId: 'your-agent-id',
 *     onStateChange: (state) => updateUI(state),
 *     onPartialTranscript: (text) => showSubtitle(text),
 *     onError: (msg) => showError(msg),
 *   });
 *
 *   await client.start();
 *   // ... later:
 *   await client.end();
 */

const BACKEND_URL   = window.location.origin;
const WS_URL        = BACKEND_URL.replace(/^http/, 'ws');
const HEARTBEAT_MS  = 30_000;   // 30s heartbeat interval
const SAMPLE_RATE   = 16000;    // target PCM sample rate
const FRAME_MS      = 20;       // VAD frame duration
const FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000;  // 320 samples per frame


export class ChronisV2Client {
  /**
   * @param {Object} options
   * @param {string} options.agentId          - Agent ID from Supabase
   * @param {string} [options.userId]         - User ID for session record
   * @param {Function} [options.onStateChange]       - (state: string) => void
   * @param {Function} [options.onPartialTranscript] - (text: string) => void
   * @param {Function} [options.onError]             - (message: string) => void
   * @param {Function} [options.onSessionEnd]        - () => void
   */
  constructor(options) {
    this.agentId   = options.agentId;
    this.userId    = options.userId || null;
    this.onState   = options.onStateChange     || (() => {});
    this.onPartial = options.onPartialTranscript || (() => {});
    this.onError   = options.onError           || console.error;
    this.onEnd     = options.onSessionEnd      || (() => {});

    this._sessionId     = null;
    this._dailyRoomUrl  = null;
    this._ws            = null;
    this._dailyCall     = null;
    this._audioCtx      = null;
    this._audioWorklet  = null;
    this._heartbeatTimer = null;
    this._active        = false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Start
  // ──────────────────────────────────────────────────────────────────────────

  async start() {
    if (this._active) throw new Error('Session already active');

    // ── Step 1: Create session on backend ──────────────────────────────────
    console.log('[ChronisV2] Starting session...');
    const startResp = await fetch(`${BACKEND_URL}/api/v2/session/start`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ agent_id: this.agentId, user_id: this.userId }),
    });

    if (!startResp.ok) {
      const err = await startResp.json();
      throw new Error(`Session start failed: ${err.detail || err.error || startResp.status}`);
    }

    const { session_id, daily_room_url } = await startResp.json();
    this._sessionId    = session_id;
    this._dailyRoomUrl = daily_room_url;
    this._active       = true;

    console.log(`[ChronisV2] Session: ${session_id.slice(0, 8)} | Room: ${daily_room_url}`);

    // ── Step 2: Open WebSocket to backend ─────────────────────────────────
    await this._connectWebSocket();

    // ── Step 3: Join Daily.co room and start audio capture ────────────────
    await this._startDailyAudio();

    // ── Step 4: Start heartbeat ────────────────────────────────────────────
    this._heartbeatTimer = setInterval(() => this._sendHeartbeat(), HEARTBEAT_MS);

    console.log('[ChronisV2] Live ✓');
  }

  // ──────────────────────────────────────────────────────────────────────────
  // End
  // ──────────────────────────────────────────────────────────────────────────

  async end() {
    if (!this._active) return;
    this._active = false;

    // Stop heartbeat
    clearInterval(this._heartbeatTimer);

    // Notify backend
    if (this._sessionId) {
      try {
        await fetch(`${BACKEND_URL}/api/v2/session/end`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ session_id: this._sessionId }),
        });
      } catch (_) {}
    }

    // Close WebSocket
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'end_session' }));
      this._ws.close();
    }

    // Stop Daily.co
    if (this._dailyCall) {
      try { await this._dailyCall.destroy(); } catch (_) {}
      this._dailyCall = null;
    }

    // Close AudioContext
    if (this._audioCtx) {
      try { await this._audioCtx.close(); } catch (_) {}
      this._audioCtx = null;
    }

    this.onEnd();
    console.log('[ChronisV2] Session ended ✓');
  }

  // ──────────────────────────────────────────────────────────────────────────
  // WebSocket
  // ──────────────────────────────────────────────────────────────────────────

  _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const url = `${WS_URL}/ws/session/${this._sessionId}`;
      console.log(`[ChronisV2] Connecting WS: ${url}`);

      this._ws = new WebSocket(url);
      this._ws.binaryType = 'arraybuffer';

      this._ws.onopen = () => {
        console.log('[ChronisV2] WebSocket connected ✓');
        resolve();
      };

      this._ws.onerror = (e) => {
        console.error('[ChronisV2] WebSocket error', e);
        reject(new Error('WebSocket connection failed'));
      };

      this._ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          this._handleServerMessage(JSON.parse(event.data));
        }
      };

      this._ws.onclose = (e) => {
        console.log(`[ChronisV2] WebSocket closed: code=${e.code} reason=${e.reason}`);
        if (this._active) {
          // Unexpected close — try to end gracefully
          this.end();
        }
      };
    });
  }

  _handleServerMessage(msg) {
    switch (msg.type) {
      case 'state':
        // Session state changed (listening/thinking/speaking/interrupted)
        this.onState(msg.state);
        break;

      case 'partial_transcript':
        // Live transcript of user speech
        this.onPartial(msg.text);
        break;

      case 'session_end':
        // Backend ended the session (watchdog or explicit end)
        console.log('[ChronisV2] Session ended by server');
        this._active = false;
        this.onEnd();
        break;

      case 'error':
        this.onError(msg.message);
        break;

      default:
        console.log('[ChronisV2] Unknown message:', msg);
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Daily.co audio capture + PCM extraction
  // ──────────────────────────────────────────────────────────────────────────

  async _startDailyAudio() {
    // ── Load Daily.co JS SDK ───────────────────────────────────────────────
    if (!window.DailyIframe) {
      await this._loadScript('https://unpkg.com/@daily-co/daily-js');
    }

    // ── Join the room ─────────────────────────────────────────────────────
    // We join as an observer with audio only — no video from our side
    this._dailyCall = window.DailyIframe.createCallObject({
      url:        this._dailyRoomUrl,
      audioSource: true,   // capture mic
      videoSource: false,  // no video from us
    });

    await this._dailyCall.join();
    console.log('[ChronisV2] Daily.co room joined ✓');

    // ── Extract the local mic MediaStreamTrack ────────────────────────────
    const tracks = this._dailyCall.participants().local.tracks;
    const audioTrack = tracks.audio?.persistentTrack;

    if (!audioTrack) {
      throw new Error('No audio track from Daily.co — check mic permissions');
    }

    // ── Set up Web Audio API pipeline ─────────────────────────────────────
    // AudioContext → ScriptProcessorNode → PCM extraction → WebSocket
    // Create AudioContext at the NATIVE browser sample rate (often 44100 or 48000).
    // Forcing { sampleRate: 16000 } is not guaranteed — browsers may silently
    // ignore it and run at 48kHz, making all our "20ms frame" math wrong.
    // Instead, we create at native rate and downsample to 16kHz in the processor.
    this._audioCtx = new AudioContext();
    const nativeSampleRate = this._audioCtx.sampleRate;
    console.log(`[ChronisV2] AudioContext native rate: ${nativeSampleRate}Hz`);
    const source   = this._audioCtx.createMediaStreamSource(
      new MediaStream([audioTrack])
    );

    // ScriptProcessorNode is deprecated but widely supported.
    // For production: replace with AudioWorkletNode (see comments below).
    //
    // Buffer size 2048 = ~128ms of audio at 16kHz.
    // We accumulate samples and send in FRAME_SAMPLES (320) chunks
    // so the backend receives consistent 20ms frames for webrtcvad.
    // Buffer size in native samples. 4096 at 48kHz ≈ 85ms per callback.
    const processor = this._audioCtx.createScriptProcessor(
      4096,  // native sample buffer
      1,     // mono input
      1,     // mono output
    );

    // How many native samples equal 20ms at the actual sample rate
    const nativeFrameSamples = Math.round(nativeSampleRate * FRAME_MS / 1000);
    // Downsampling ratio: native → 16kHz
    const downsampleRatio    = nativeSampleRate / SAMPLE_RATE;  // e.g. 3.0 at 48kHz

    let sampleBuffer = new Float32Array(0);

    processor.onaudioprocess = (event) => {
      if (!this._active || !this._ws || this._ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const float32 = event.inputBuffer.getChannelData(0);

      // Accumulate at native rate
      const combined = new Float32Array(sampleBuffer.length + float32.length);
      combined.set(sampleBuffer);
      combined.set(float32, sampleBuffer.length);
      sampleBuffer = combined;

      // Process in native 20ms frames, then downsample each to 16kHz
      while (sampleBuffer.length >= nativeFrameSamples) {
        const nativeFrame = sampleBuffer.slice(0, nativeFrameSamples);
        sampleBuffer      = sampleBuffer.slice(nativeFrameSamples);

        // Downsample to 16kHz via linear interpolation
        const targetLength  = FRAME_SAMPLES;  // 320 samples at 16kHz
        const downsampled   = new Float32Array(targetLength);
        for (let i = 0; i < targetLength; i++) {
          const srcIdx    = i * downsampleRatio;
          const lo        = Math.floor(srcIdx);
          const hi        = Math.min(lo + 1, nativeFrame.length - 1);
          const frac      = srcIdx - lo;
          downsampled[i]  = nativeFrame[lo] * (1 - frac) + nativeFrame[hi] * frac;
        }

        const pcm = this._float32ToPCM16(downsampled);
        this._ws.send(pcm.buffer);
      }
    };

    // Connect source → processor only (NOT to destination — avoids mic loopback/echo)
    source.connect(processor);
    // Deliberately do NOT connect processor to destination

    console.log('[ChronisV2] Audio capture started ✓ (16kHz mono PCM)');

    /*
     * PRODUCTION UPGRADE: AudioWorklet version
     * Replace the ScriptProcessorNode with an AudioWorkletNode for
     * better timing precision and no garbage collection jank.
     *
     * 1. Create worklet file: static/js/pcm_processor.worklet.js
     *    class PCMProcessor extends AudioWorkletProcessor {
     *      process(inputs) {
     *        const frame = inputs[0][0];
     *        this.port.postMessage({ frame }, [frame.buffer]);
     *        return true;
     *      }
     *    }
     *    registerProcessor('pcm-processor', PCMProcessor);
     *
     * 2. In _startDailyAudio():
     *    await this._audioCtx.audioWorklet.addModule('/js/pcm_processor.worklet.js');
     *    const workletNode = new AudioWorkletNode(this._audioCtx, 'pcm-processor');
     *    workletNode.port.onmessage = (e) => {
     *      const pcm = this._float32ToPCM16(e.data.frame);
     *      this._ws.send(pcm.buffer);
     *    };
     *    source.connect(workletNode);
     */
  }

  _float32ToPCM16(float32Array) {
    /**
     * Convert Float32 audio [-1.0, 1.0] to Int16 PCM.
     * This is what webrtcvad and Deepgram expect.
     */
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s    = Math.max(-1, Math.min(1, float32Array[i]));
      int16[i]   = s < 0 ? s * 32768 : s * 32767;
    }
    return int16;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Heartbeat
  // ──────────────────────────────────────────────────────────────────────────

  async _sendHeartbeat() {
    if (!this._active || !this._sessionId) return;
    try {
      await fetch(`${BACKEND_URL}/api/v2/session/heartbeat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: this._sessionId }),
      });
      // Also send via WebSocket in case fetch is slow
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'heartbeat' }));
      }
    } catch (_) {
      // Heartbeat failure is non-fatal — watchdog will eventually clean up
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Utilities
  // ──────────────────────────────────────────────────────────────────────────

  _loadScript(src) {
    return new Promise((resolve, reject) => {
      const s    = document.createElement('script');
      s.src      = src;
      s.onload   = resolve;
      s.onerror  = reject;
      document.head.appendChild(s);
    });
  }

  get sessionId()    { return this._sessionId; }
  get dailyRoomUrl() { return this._dailyRoomUrl; }   // public getter — live.html uses this
  get isActive()     { return this._active; }
}


/**
 * ── USAGE EXAMPLE ──────────────────────────────────────────────────────────
 *
 * <button id="startBtn">Start Conversation</button>
 * <button id="endBtn" disabled>End</button>
 * <div id="state">idle</div>
 * <div id="transcript"></div>
 *
 * <script type="module">
 *   import { ChronisV2Client } from './chronis_v2_client.js';
 *
 *   const client = new ChronisV2Client({
 *     agentId: 'your-agent-uuid',
 *     onStateChange: (state) => {
 *       document.getElementById('state').textContent = state;
 *       document.getElementById('state').className = `state-${state}`;
 *     },
 *     onPartialTranscript: (text) => {
 *       document.getElementById('transcript').textContent = text;
 *     },
 *     onError: (msg) => alert(`Error: ${msg}`),
 *     onSessionEnd: () => {
 *       document.getElementById('startBtn').disabled = false;
 *       document.getElementById('endBtn').disabled = true;
 *     },
 *   });
 *
 *   document.getElementById('startBtn').onclick = async () => {
 *     document.getElementById('startBtn').disabled = true;
 *     document.getElementById('endBtn').disabled = false;
 *     await client.start();
 *   };
 *
 *   document.getElementById('endBtn').onclick = () => client.end();
 * </script>
 */
