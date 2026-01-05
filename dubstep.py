import numpy as np
import simpleaudio as sa
from pydub import AudioSegment
from gtts import gTTS
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, jsonify, render_template_string
from pydub.effects import low_pass_filter, normalize, high_pass_filter
import webbrowser
import threading
import time

# Global thread lock
lock = threading.Lock()

# Flask app for live plotting
app = Flask(__name__)

# Globals used by the server
start_time = None
# sample rate of the audio buffer (set in main)
audio_sample_rate = 44100
# subtitle globals
subtitle_words = []
word_duration_ms = 0.0
voice_offset_ms = 10000  # we overlay voice at 10000 ms (10 seconds)
# amplitude range for plotting
audio_max_amp = 1.0


pio.renderers.default = 'browser'

TEXT_TO_SPEECH = (
    "Der anarchistische Herrscher sitzend wie ein blinder König auf seinen Thron. Es freut mich "
    "vorzüglich sie kennen zu lernen Herr Bashir, fühlen sie sich gegrüßt. Der Staat ist eine Abstraktion, "
    "die das Leben des Volkes verschlingt. Ein unermesslicher Friedhof, auf dem alle Lebenskräfte eines "
    "Landes großzügig und andächtig sich haben hinschlachten lassen. Denn die Lust der Zerstörung ist "
    "zugleich eine Schaffende Lust, lasst uns also dem ewigen Geiste vertrauen, der nur deshalb zerstört "
    "und vernichtet, weil er der unergründliche und ewig schaffende Quell allen Lebens ist. So denken sie "
    "doch nicht liebster Sheik? Was mich allerdings betrifft, so glaube ich nicht, dass es eine Lösung für "
    "die gesellschaftlichen Probleme gibt, sondern tausend verschiedene und veränderbare Lösungen, wie "
    "auch das gesellschaftliche Leben in Zeit und Raum verschieden und veränderbar ist! Allerdings stellt "
    "sich mir da noch die eine Frage. "
)

def generate_voice(text, filename="voice.wav"):
    try:
        tts = gTTS(text=text, lang='de')
        tts.save("voice.mp3")
        audio_segment = AudioSegment.from_mp3("voice.mp3")
        # Normalize voice segment to match track parameters
        audio_segment = audio_segment.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio_segment.export(filename, format="wav")
        return audio_segment
    except Exception as e:
        print(f"Error generating voice: {e}")
        return None

def generate_wave(frequency, duration, wave_type='sine'):
    t = np.linspace(0, duration / 1000, int(44100 * duration / 1000), endpoint=False)
    # Improved generator: supports sine, saw, square, noise, wobble with ADSR
    sample_rate = 44100
    n = int(sample_rate * (duration / 1000.0))
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)

    if wave_type == 'sine':
        wave = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'saw':
        # naive saw using fractional part
        frac = (frequency * t) - np.floor(frequency * t)
        wave = 2.0 * frac - 1.0
    elif wave_type == 'square':
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'noise':
        wave = np.random.normal(0, 0.3, t.shape)
    elif wave_type == 'wobble':
        # wobble: frequency modulated by low-frequency oscillator (LFO)
        lfo_freq = 2.5
        # limit wobble depth to avoid huge frequency excursions (keeps tone stable)
        depth = min(max(0.5, frequency * 0.05), 8.0)
        mod = np.sin(2 * np.pi * lfo_freq * t) * depth
        wave = np.sin(2 * np.pi * (frequency + mod) * t)
    else:
        wave = np.sin(2 * np.pi * frequency * t)

    # Simple ADSR envelope: attack 5ms, decay 50ms, sustain 0.7, release 100ms or remainder
    attack = int(sample_rate * 0.005)
    decay = int(sample_rate * 0.05)
    release = int(sample_rate * 0.1)
    sustain_level = 0.7
    env = np.ones(n)
    if n > 0:
        # attack
        a_end = min(attack, n)
        if a_end > 0:
            env[:a_end] = np.linspace(0, 1.0, a_end)
        # decay
        d_end = min(a_end + decay, n)
        if d_end > a_end:
            env[a_end:d_end] = np.linspace(1.0, sustain_level, d_end - a_end)
        # release
        if n > (a_end + decay + release):
            r_start = n - release
            env[r_start:] = np.linspace(sustain_level, 0.0, n - r_start)
        else:
            # short sounds: ramp down to 0
            env[d_end:] = np.linspace(sustain_level, 0.0, n - d_end)

    wave = wave * env
    # apply light decay for longer sounds
    wave *= np.exp(-t / (duration / 2000.0))
    # normalize to int16
    max_val = np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1.0
    wave = wave / max_val
    return (wave * 32767).astype(np.int16)


def generate_kick(duration=300, sample_rate=44100):
    # Punchy kick with pitch drop, sub reinforcement and transient click
    start_freq = 150.0
    end_freq = 40.0
    n = int(sample_rate * (duration / 1000.0))
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)
    # exponential pitch drop for tonal body
    freq = start_freq * (end_freq / start_freq) ** (t / t[-1])
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    body = np.sin(phase)

    # Sub reinforcement (low sine) to add fullness
    sub_freq = 50.0
    sub = np.sin(2 * np.pi * sub_freq * t) * np.exp(-3 * t)

    # transient click (high-frequency) for attack
    click_len = min(200, n)
    click = np.zeros(n)
    if click_len > 0:
        transient = np.random.normal(0, 1.0, click_len) * np.linspace(1.0, 0.0, click_len)
        # high-pass the transient by shaping with a short sine burst
        click[:click_len] = transient * 0.6

    # amplitude envelope: quick attack and decay for body
    env = np.exp(-6 * t)
    wave = (0.9 * body + 0.7 * sub + 0.6 * click) * env
    # gentle soft-clip to add perceived loudness
    wave = np.tanh(wave * 2.0)
    max_val = np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1.0
    wave = wave / max_val
    return (wave * 32767).astype(np.int16)


def generate_bass(frequency, duration, sample_rate=44100):
    """Generate a dubstep-style bass: sub sine + wobble layer with soft clipping."""
    # sub sine (pure low end)
    n = int(sample_rate * (duration / 1000.0))
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)
    sub = np.sin(2 * np.pi * frequency * t)

    # wobble layer (richer harmonics)
    wobble = generate_wave(frequency, duration, wave_type='wobble').astype(np.float32) / 32767.0

    # mix sub with wobble (wobble quieter to avoid harsh upper harmonics)
    mix = 0.85 * sub + 0.5 * wobble

    # apply soft clipping (drive)
    drive = 2.0
    mixed = np.tanh(mix * drive)

    # lowpass-ish decay by multiplying a slow envelope to avoid too long tails
    env = np.exp(-t * 0.5)
    mixed *= env

    # normalize
    maxv = np.max(np.abs(mixed)) if np.max(np.abs(mixed)) > 0 else 1.0
    mixed = mixed / maxv
    return (mixed * 32767).astype(np.int16)


def generate_hihat(duration=80, sample_rate=44100):
    # short filtered noise burst
    n = int(sample_rate * (duration / 1000.0))
    # lower the noise amplitude to reduce hiss
    noise = np.random.normal(0, 0.35, n)
    # light envelope
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)
    env = np.exp(-30 * t)
    wave = noise * env
    maxv = np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1.0
    wave = wave / maxv
    arr = (wave * 32767).astype(np.int16)
    seg = AudioSegment(arr.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    # remove low frequencies from hats to prevent mud
    try:
        seg = high_pass_filter(seg, cutoff=2500)
    except Exception:
        pass
    return np.array(seg.get_array_of_samples()).astype(np.int16)


def generate_chord(frequencies, duration, wave_type='saw'):
    """Generate a chord by summing several generator waves and normalizing."""
    if wave_type == 'piano':
        parts = [generate_piano(f, duration).astype(np.float32) for f in frequencies]
    elif wave_type == 'piano_violin':
        # create piano attack layer and sustained violin layer, then blend
        piano_parts = [generate_piano(f, duration).astype(np.float32) for f in frequencies]
        violin_parts = [generate_violin(f, duration).astype(np.float32) for f in frequencies]
        # sum piano and violin separately then mix
        piano_sum = np.sum(np.vstack([p[:min_len] if (min_len := min(p.shape[0] for p in piano_parts)) else p for p in piano_parts]), axis=0)
        violin_sum = np.sum(np.vstack([v[:min_len] if (min_len := min(v.shape[0] for v in violin_parts)) else v for v in violin_parts]), axis=0)
        # adjust relative levels: piano gives attack, violin gives sustain
        # convert to same length and mix
        min_len = min(piano_sum.shape[0], violin_sum.shape[0])
        mixed = 0.6 * piano_sum[:min_len] + 1.0 * violin_sum[:min_len]
        maxv = np.max(np.abs(mixed)) if np.max(np.abs(mixed)) > 0 else 1.0
        chord = (mixed / maxv * 32767).astype(np.int16)
        return chord
    else:
        parts = [generate_wave(f, duration, wave_type=wave_type).astype(np.float32) for f in frequencies]
    # pad/truncate to same length if necessary
    min_len = min(p.shape[0] for p in parts)
    parts = [p[:min_len] for p in parts]
    summed = np.sum(np.vstack(parts), axis=0)
    maxv = np.max(np.abs(summed)) if np.max(np.abs(summed)) > 0 else 1.0
    chord = (summed / maxv * 32767).astype(np.int16)
    return chord


def generate_violin(frequency, duration, sample_rate=44100):
    """Simple bowed-string (violin-like) synthesis with vibrato and bow noise."""
    n = int(sample_rate * (duration / 1000.0))
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)

    # base harmonic content (saw-like richness but filtered by envelope)
    harmonics = [1.0, 0.8, 0.6, 0.4, 0.25]
    tone = np.zeros(n, dtype=np.float32)
    for i, h in enumerate(harmonics):
        # slight inharmonicity for realism
        freq = frequency * (i + 1) * (1.0 + 0.0008 * i)
        tone += h * np.sin(2 * np.pi * freq * t)

    # Bow noise component (narrow-band noise shaped by envelope) - reduced level for cleanliness
    noise = np.random.normal(0, 0.4, n)
    noise *= np.exp(-t * 1.2)
    tone += 0.06 * noise

    # Vibrato (slow frequency modulation)
    vibrato_rate = 5.0
    vibrato_depth = 0.003 * frequency
    vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
    # apply vibrato by resampling phase (approximate)
    phase = 2 * np.pi * np.cumsum((frequency + vibrato) / sample_rate)
    violin_wave = np.sin(phase)
    # combine with tone and apply long sustain envelope
    env = np.ones(n)
    attack = int(sample_rate * 0.02)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    # slow decay to simulate bowed sustain
    if n > attack:
        env[attack:] = np.linspace(1.0, 0.5, n - attack)

    out = (0.8 * tone + 0.9 * violin_wave) * env
    # gentle brightness roll-off
    out *= np.exp(-t * 0.3)
    maxv = np.max(np.abs(out)) if np.max(np.abs(out)) > 0 else 1.0
    out = out / maxv
    return (out * 32767).astype(np.int16)


def generate_piano(frequency, duration, sample_rate=44100):
    """Synthesize a simple piano-like tone using additive synthesis and a hammer transient.
    Not a sampled piano, but a convincing percussive harmonic tone suitable for chords.
    Returns a numpy int16 array.
    """
    n = int(sample_rate * (duration / 1000.0))
    t = np.linspace(0, duration / 1000.0, n, endpoint=False)

    # harmonic partials and relative amplitudes (decay faster for higher partials)
    partials = [1.0, 0.58, 0.34, 0.22, 0.12, 0.06]
    decay = 4.0  # overall decay rate (seconds^-1)
    tone = np.zeros(n, dtype=np.float32)
    for i, amp in enumerate(partials):
        partial_freq = frequency * (i + 1)
        # slight inharmonicity for realism
        inharm = partial_freq * (1.0 + 0.0005 * (i))
        tone += amp * np.sin(2 * np.pi * inharm * t) * np.exp(-decay * (i + 1) * t)

    # hammer transient: short filtered noise burst at the start
    transient_len = int(min(n, int(sample_rate * 0.006)))
    if transient_len > 0:
        transient = np.random.normal(0, 1.0, transient_len) * np.linspace(1.0, 0.0, transient_len)
        # highpass-ish shaping via subtracting a low-frequency component
        transient = transient - np.mean(transient) * 0.3
        tone[:transient_len] += transient * 0.8

    # apply per-note envelope: very fast attack, medium decay, gentle release
    env = np.ones(n)
    attack_samples = int(sample_rate * 0.0015)
    decay_samples = int(sample_rate * 0.5)
    release_samples = int(sample_rate * 0.5)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1.0, attack_samples)
    if n > attack_samples:
        d_end = min(n, attack_samples + decay_samples)
        env[attack_samples:d_end] = np.linspace(1.0, 0.6, d_end - attack_samples)
        if n > d_end + release_samples:
            r_start = n - release_samples
            env[r_start:] = np.linspace(0.6, 0.0, n - r_start)
        else:
            env[d_end:] = np.linspace(0.6, 0.0, n - d_end)

    tone *= env

    # gentle brightness control
    tone *= np.exp(-t * 0.6)

    # normalize and return int16
    maxv = np.max(np.abs(tone)) if np.max(np.abs(tone)) > 0 else 1.0
    tone = tone / maxv
    return (tone * 32767).astype(np.int16)


def generate_arp_note(frequency, dur_ms=150, wave_type='sine'):
    """Generate a short arpeggio note as AudioSegment."""
    arr = generate_wave(frequency, dur_ms, wave_type=wave_type).astype(np.int16)
    seg = AudioSegment(arr.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    return seg.set_channels(2)


def generate_fill(duration_ms=600, sample_rate=44100):
    """Generate a short percussion/noise fill to drop in at transitions."""
    n = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n, endpoint=False)
    noise = np.random.normal(0, 0.6, n)
    env = np.exp(-6.0 * t)
    wave = noise * env
    # gentle highpass-like shaping by subtracting a low-freq sine component
    wave = wave - 0.2 * np.sin(2 * np.pi * 50.0 * t)
    maxv = np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1.0
    wave = wave / maxv
    arr = (wave * 32767).astype(np.int16)
    seg = AudioSegment(arr.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    # add brightness and cut lows
    try:
        # cut some low end to keep the fill from muddying chords
        seg = high_pass_filter(seg, cutoff=400)
    except Exception:
        pass
    return seg.set_channels(2)


def soft_clip_segment(seg, drive=1.05):
    """Apply a gentle soft-clip limiter to an AudioSegment using tanh on normalized samples."""
    try:
        arr = np.array(seg.get_array_of_samples())
        # handle stereo
        channels = seg.channels
        if channels == 2:
            arr = arr.reshape((-1, 2)).astype(np.float32)
            maxv = np.max(np.abs(arr)) if np.max(np.abs(arr)) > 0 else 1.0
            arr = arr / maxv
            arr = np.tanh(arr * drive)
            out = (arr * 32767.0).astype(np.int16)
            return AudioSegment(out.tobytes(), frame_rate=seg.frame_rate, sample_width=2, channels=2)
        else:
            arr = arr.astype(np.float32)
            maxv = np.max(np.abs(arr)) if np.max(np.abs(arr)) > 0 else 1.0
            arr = arr / maxv
            arr = np.tanh(arr * drive)
            out = (arr * 32767.0).astype(np.int16)
            return AudioSegment(out.tobytes(), frame_rate=seg.frame_rate, sample_width=2, channels=1)
    except Exception:
        return seg


def add_reverb_delay(seg, delay_ms=120, repeats=3, decay_dB=6, lowpass_cut=5000):
    """Simple slap-delay + filtered repeats to give a sense of space without muddying lows."""
    out = seg
    for i in range(1, repeats + 1):
        copy = seg - (decay_dB * i)
        try:
            copy = low_pass_filter(copy, lowpass_cut)
        except Exception:
            pass
        out = out.overlay(copy, position=delay_ms * i)
    return out


def apply_mastering(seg):
    """Mastering chain: gentle HPF to remove rumble, subtle delay/reverb, soft clip, and final normalize."""
    s = seg
    # remove infrasonic rumble
    try:
        s = high_pass_filter(s, cutoff=30)
    except Exception:
        pass
    # simple multiband processing: saturate mids and boost presence
    try:
        # split bands
        low = low_pass_filter(s, cutoff=120)
        mid = high_pass_filter(s, cutoff=120)
        mid = low_pass_filter(mid, cutoff=3000)
        high = high_pass_filter(s, cutoff=3000)

        # process mids: gentle saturation
        mid = soft_clip_segment(mid, drive=1.15)
        mid = mid.apply_gain(2.0)

        # process highs: add presence
        high = high.apply_gain(2.5)

        # recombine
        combined = low.overlay(mid).overlay(high)
        s = combined
    except Exception:
        pass

    # add gentle ambience/delay to glue elements (after band work)
    try:
        # reduce repeats/decay to avoid buildup of noise
        s = add_reverb_delay(s, delay_ms=80, repeats=2, decay_dB=6, lowpass_cut=4000)
    except Exception:
        pass

    # stronger soft limiting to control peaks and increase loudness
    try:
        s = soft_clip_segment(s, drive=1.12)
    except Exception:
        pass

    # final target loudness for commercial release
    try:
        target_dbfs = -1.0
        change = target_dbfs - s.dBFS
        s = s.apply_gain(change)
    except Exception:
        try:
            s = normalize(s)
        except Exception:
            pass
    # final gentle low-pass to remove ultrasonic artifacts and residual hiss
    try:
        s = low_pass_filter(s, cutoff=14000)
    except Exception:
        pass
    return s

def create_dubstep_track(voice_segment):
    # Build richer stereo layers
    # Bass (wobble), kick, hihat (noise), synth (saw)
    # Core elements
    bass_wave = generate_bass(55, 8000)
    kick_wave = generate_kick(300)
    hihat_wave = generate_hihat(80)
    synth_wave = generate_wave(220, 1000, wave_type='saw')

    bass_seg = AudioSegment(bass_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    kick_seg = AudioSegment(kick_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    hihat_seg = AudioSegment(hihat_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    synth_seg = AudioSegment(synth_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)

    # Convert to stereo and apply panning
    bass_st = bass_seg.set_channels(2).pan(0.0)        # centered
    kick_st = kick_seg.set_channels(2).pan(0.0)        # centered
    hihat_st = hihat_seg.set_channels(2).pan(0.6)      # slightly right
    synth_left = synth_seg.set_channels(2).pan(-0.5)
    synth_right = synth_seg.set_channels(2).pan(0.5)

    # Create silent stereo base track
    track = AudioSegment.silent(duration=120000, frame_rate=44100).set_channels(2)  # 2 minutes stereo

    # Use musical timing (BPM) for better dubstep grooves
    BPM = 140
    beat_ms = int(60000 / BPM)  # quarter note duration in ms

    # First pass: add kick, hihats, synths and record kick times for ducking
    kick_times = []
    for i in range(0, int(120000 / beat_ms)):
        ms = i * beat_ms
        # Kick on every beat
        track = track.overlay(kick_st, position=ms)
        kick_times.append(ms)

        # Hihat on 8th notes (twice per beat)
        eighth = beat_ms // 2
        track = track.overlay(hihat_st - 6, position=ms)
        track = track.overlay(hihat_st - 9, position=ms + eighth)

        # Synth accents every 4 beats
        if i % 4 == 0:
            track = track.overlay(synth_left - 8, position=ms)
            track = track.overlay(synth_right - 8, position=ms)

        # Evolving complexity: increase density and add arps/fills over time
        section_ms = 15000
        section = int(ms // section_ms)
        # section 0: base
        # section 1+: add extra hats and slight variations
        if section >= 1:
            # extra off-beat hat
            off = beat_ms // 4
            track = track.overlay(hihat_st - 10, position=ms + off)
            track = track.overlay(hihat_st - 12, position=ms + off * 3)

        # section 2+: introduce arpeggios / fast synth lines
        if section >= 2:
            # pick a simple arp pattern based on root 220Hz shifting
            arp_root = 220.0 + (section - 2) * 20.0
            arp_notes = [arp_root, arp_root * 1.5, arp_root * 2.0, arp_root * 2.5]
            step = beat_ms // 4
            for j, f in enumerate(arp_notes):
                note_pos = ms + j * step
                arp_seg = generate_arp_note(f, dur_ms=step, wave_type='sine') - 14
                pan_val = -0.4 + 0.2 * (j % 3)
                track = track.overlay(arp_seg.pan(pan_val), position=note_pos)

        # section 3+: add fills and extra percussion around bar ends
        if section >= 3:
            # occasionally drop a fill at the end of a 4-beat bar
            if i % 4 == 3:
                fill = generate_fill(duration_ms=int(beat_ms * 0.9)) - 10
                track = track.overlay(fill, position=max(0, ms - int(beat_ms * 0.5)))

    # Second pass: add bass with filter-LFO processing and ducking around kick times
    # Prepare base bass segment (mono)
    bass_seg = AudioSegment(bass_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)

    def process_bass_for_position(bass_seg, start_pos_ms, lfo_freq=2.0, cut_low=100, cut_high=2000, chunk_ms=60):
        # Process bass by chopping into chunks and applying a time-varying low-pass filter
        duration_ms = len(bass_seg)
        out = AudioSegment.silent(duration=0, frame_rate=bass_seg.frame_rate)
        for off in range(0, duration_ms, chunk_ms):
            chunk = bass_seg[off:off+chunk_ms]
            # compute absolute time in ms for this chunk
            t_ms = start_pos_ms + off
            t_sec = t_ms / 1000.0
            # LFO value
            lfo = 0.5 * (1.0 + np.sin(2 * np.pi * lfo_freq * t_sec))
            cutoff = int(cut_low + (cut_high - cut_low) * lfo)
            try:
                filtered = low_pass_filter(chunk, cutoff)
            except Exception:
                filtered = chunk
            out += filtered
        return out

    # Overlay bass on half-note positions with processing and ducking
    half_note_ms = beat_ms * 2
    for pos in range(0, 120000, half_note_ms):
        # create processed bass for this position
        processed = process_bass_for_position(bass_seg, pos, lfo_freq=1.5, cut_low=80, cut_high=1200, chunk_ms=80)
        # apply stereo and center
        processed = processed.set_channels(2).pan(0.0) - 4
        # Duck bass around kicks: attenuate bass for a short window after each kick
        for k in kick_times:
            if k <= pos + len(processed) and k >= pos - 200:
                # compute overlap region within processed
                rel = k - pos
                start = max(0, rel)
                end = min(len(processed), rel + int(beat_ms * 0.35))
                if start < end:
                    head = processed[:start]
                    mid = (processed[start:end] - 12)
                    tail = processed[end:]
                    processed = head + mid + tail
        track = track.overlay(processed, position=pos)

    # Add a dark pad starting at 5 seconds (5000 ms)
    pad_roots = [55.0, 49.0, 58.27]  # low roots for dark atmosphere
    pad_dur = 8000
    pos = 5000
    while pos < 120000:
        root = pad_roots[((pos - 5000) // pad_dur) % len(pad_roots)]
        freqs = [root, root * (6.0/5.0), root * (3.0/2.0)]
        # Create two slightly detuned layers for thickness
        chord_wave = generate_chord(freqs, pad_dur, wave_type='saw')
        chord_wave2 = generate_chord([f * 1.01 for f in freqs], pad_dur, wave_type='saw')
        summed = ((chord_wave.astype(np.int32) + chord_wave2.astype(np.int32)) // 2).astype(np.int16)
        chord_seg = AudioSegment(summed.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        # Darken with low-pass and reduce level
        chord_seg = low_pass_filter(chord_seg, cutoff=1000)
        chord_seg = chord_seg - 10
        chord_st = chord_seg.set_channels(2).pan(-0.2)
        track = track.overlay(chord_st, position=pos)
        pos += pad_dur

    # Add chord progression starting at 30 seconds (30000 ms)
    # simple progression (root, minor third, fifth) shifting roots for variety
    chord_roots = [110, 98, 123.47, 82.41]  # A, G, B, E (approx)
    chord_dur = 4000
    for i, root in enumerate(chord_roots):
        pos = 30000 + i * chord_dur
        if pos >= 120000:
            break
        freqs = [root, root * (6/5), root * (3/2)]
        chord_wave = generate_chord(freqs, chord_dur, wave_type='saw')
        chord_seg = AudioSegment(chord_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        chord_st = chord_seg.set_channels(2).pan(0.0)
        track = track.overlay(chord_st, position=pos)

    # Overlay voice (ensure voice_segment is stereo-matched)
    voice_stereo = voice_segment.set_frame_rate(44100).set_channels(2).set_sample_width(2)
    result = track.overlay(voice_stereo, position=voice_offset_ms)
    # Apply a small mastering chain to clean up and make the track more commercial-sounding
    try:
        result = apply_mastering(result)
    except Exception:
        try:
            # Fallback to basic normalize if mastering fails
            result = normalize(result)
        except Exception:
            pass
    return result

def play_audio(track):
    # Play audio using simpleaudio
    global start_time
    start_time = time.time()
    sa.play_buffer(track.raw_data, num_channels=track.channels, bytes_per_sample=track.sample_width, sample_rate=track.frame_rate).wait_done()
    print("Audio playback completed.")


def update_plot():
    global audio_samples, idx, fig
    while idx < len(audio_samples):
        with lock:  # Ensure thread-safe access to idx
            current_samples = audio_samples[idx:idx + 1000]
            length = current_samples.size

            # Check if there's actual data to plot
            if length == 0:
                print("No samples to display.")
                break

            time_axis = np.arange(idx, idx + length) / 44100  # Convert samples to seconds

            # Update the figure's data
            fig.data[0].update(x=time_axis, y=current_samples)

            # Force the figure to update
            fig.update_traces()

            idx += length
        
        time.sleep(0.1)



def create_interactive_plot():
    # Kept for backwards compatibility if live updates are added later
    # Prefer using `plot_full_waveform` to display the complete waveform in the browser.
    fig = go.Figure()
    return fig


def plot_full_waveform(samples, sample_rate=44100):
    """Plot the full waveform once (browser HTML won't reflect later Python updates)."""
    # Downsample for plotting to avoid freezing the browser with millions of points
    max_points = 100000
    length = len(samples)
    if length > max_points:
        factor = int(np.ceil(length / max_points))
        samples_to_plot = samples[::factor]
        time_axis = np.arange(0, length, factor) / sample_rate
    else:
        samples_to_plot = samples
        time_axis = np.arange(length) / sample_rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=samples_to_plot, mode='lines', line=dict(color='royalblue', width=1)))
    fig.update_layout(
        title="Waveform of the Dubstep Track",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )
    fig.show()


@app.route('/')
def index():
        # Simple HTML page that polls /samples
        html = '''
        <!doctype html>
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
                <body style="background:black;color:white;margin:0;padding:0;height:100vh;">
                    <div id="plot" style="width:100%;height:100vh"></div>
                <script>
                                        const plotDiv = document.getElementById('plot');
                        // Layout will be adjusted dynamically based on server window_duration
                        let layout = {title: {text: 'Live Waveform', font: {color:'white'}}, margin:{t:40}, paper_bgcolor:'black', plot_bgcolor:'black', font:{color:'white'}, xaxis:{range:[-0.12,0], title:'Time (s)'}};
                    Plotly.newPlot(plotDiv, [{x:[], y:[], mode:'lines', line:{color:'cyan'}}], layout, {displayModeBar:false});

                    async function fetchSamples(){
                        try{
                            const res = await fetch('/samples');
                            const j = await res.json();
                            if(j.x && j.y){
                                // adjust x-axis to server window duration so waveform fills width
                                if(j.window_duration){
                                    layout.xaxis.range = [-j.window_duration, 0];
                                }
                                                                        // Use Plotly.react for faster full-frame updates
                                                                        // Set symmetric y-axis range based on current window peak so waveform fills height
                                                                        if(j.y_peak){
                                                                            const peak = Math.max(1e-6, j.y_peak);
                                                                            layout.yaxis = {range:[-peak*1.05, peak*1.05], autorange:false};
                                                                        }
                                                                        // Add a low-opacity subtitle annotation behind the waveform
                                                                        const ann = [{text: (j.subtitle||''), x:0.5, xref:'paper', xanchor:'center', y:0.02, yref:'paper', yanchor:'bottom', font:{size:28, color:'white'}, opacity:0.12, showarrow:false, align:'center', layer:'below'}];
                                                                        layout.annotations = ann;
                                                                        Plotly.react(plotDiv, [{x:j.x, y:j.y, mode:'lines', line:{color:'cyan', width:1}}], layout, {displayModeBar:false});
                            }
                        }catch(e){
                            console.log('fetch error', e);
                        }
                    }

                    // Poll faster for smoother animation (~20 Hz)
                    setInterval(fetchSamples, 50);
                </script>
            </body>
        </html>
        '''
        return render_template_string(html)


@app.route('/samples')
def samples_endpoint():
    global audio_samples, start_time, audio_sample_rate
    if start_time is None or audio_samples is None:
        return jsonify({'x':[], 'y':[]})

    # Compute current index from playback start time
    elapsed = time.time() - start_time
    current_idx = int(elapsed * audio_sample_rate)

    # Return a fixed-length window (in samples) ending at current_idx for consistent, smooth plotting
    window = 4096
    end = min(len(audio_samples), current_idx)
    start = end - window
    if start < 0:
        # Pad at the front with zeros if playback hasn't reached full window yet
        pad = -start
        start = 0
    else:
        pad = 0

    window_samples = audio_samples[start:end]

    # If no samples yet, return zeros
    if window_samples.size == 0:
        x = (np.linspace(-window/float(audio_sample_rate), 0.0, window)).tolist()
        y = ([0] * window)
        return jsonify({'x': x, 'y': y})

    # Determine peak of current window for scaling the plot
    try:
        y_peak_val = float(np.max(np.abs(window_samples))) if window_samples.size > 0 else 0.0
    except Exception:
        y_peak_val = 0.0

    # Downsample for transfer size if necessary
    max_points = 2000
    length = len(window_samples)
    if length > max_points:
        factor = int(np.ceil(length / max_points))
        y_down = window_samples[::factor].tolist()
        x_down = (np.linspace((-(length))/float(audio_sample_rate), 0.0, len(y_down))).tolist()
    else:
        y_down = window_samples.tolist()
        x_down = (np.linspace((-(length))/float(audio_sample_rate), 0.0, length)).tolist()

    # If we had to pad at the front, prepend zeros to reach full window length
    if pad > 0:
        zeros = [0] * int(np.ceil(pad/float(max(1, factor if 'factor' in locals() else 1))))
        y = zeros + y_down
        # Build x to be from -window/sample_rate .. 0
        x = (np.linspace(-window/float(audio_sample_rate), 0.0, len(y))).tolist()
    else:
        y = y_down
        x = x_down

    # Compute subtitle based on elapsed time and precomputed per-word duration
    subtitle = ""
    try:
        current_ms_total = int(elapsed * 1000.0)
        # time into the voice (account for overlay offset)
        voice_ms = current_ms_total - voice_offset_ms
        if voice_ms > 0 and len(subtitle_words) > 0 and word_duration_ms > 0:
            idx = int(voice_ms // word_duration_ms)
            if idx < 0:
                subtitle = ""
            else:
                # show a small window of words around current word
                start_w = max(0, idx - 3)
                end_w = min(len(subtitle_words), idx + 4)
                subtitle = " ".join(subtitle_words[start_w:end_w])
    except Exception:
        subtitle = ""

    # Ensure the arrays are not too large for transfer
    window_duration = float(window) / float(audio_sample_rate)
    # Provide y-axis amplitude info so client can set symmetric range
    global audio_max_amp
    # Use peak of current window if available, otherwise fall back to global max
    y_peak = float(y_peak_val) if y_peak_val > 0 else float(audio_max_amp)
    return jsonify({'x': x, 'y': y, 'window_duration': window_duration, 'subtitle': subtitle, 'y_max': float(audio_max_amp), 'y_peak': y_peak})


def start_plot_server(host='127.0.0.1', port=8050):
    # Run Flask server in a separate thread (development server)
    server_thread = threading.Thread(target=app.run, kwargs={'host':host, 'port':port, 'threaded':True, 'use_reloader':False}, daemon=True)
    server_thread.start()
    print(f"Plot server running at http://{host}:{port}/")



def main():
    global audio_samples, idx  # Don't need ax here
    voice_segment = generate_voice(TEXT_TO_SPEECH)
    
    if voice_segment is not None:
        track = create_dubstep_track(voice_segment)
        # Use mono mixdown for plotting and analysis (track remains stereo for playback)
        audio_samples = np.array(track.set_channels(1).get_array_of_samples())
        if audio_samples.size == 0:
            print("Error: No audio samples generated.")
            return
        
        idx = 0
        # Set sample rate for the live samples endpoint
        global audio_sample_rate
        audio_sample_rate = track.frame_rate
        # Set global amplitude range for plotting (use full track samples)
        global audio_max_amp
        try:
            audio_max_amp = float(np.max(np.abs(audio_samples)))
            if audio_max_amp <= 0:
                audio_max_amp = 1.0
        except Exception:
            audio_max_amp = 1.0
        # Prepare subtitle timing: split text into words and compute per-word duration (ms)
        global subtitle_words, word_duration_ms
        subtitle_words = TEXT_TO_SPEECH.split()
        if len(subtitle_words) > 0:
            word_duration_ms = float(len(voice_segment)) / float(len(subtitle_words))
        else:
            word_duration_ms = 0.0

        # Start the live plot server and open the browser
        start_plot_server(host='127.0.0.1', port=8050)
        try:
            webbrowser.open(f'http://127.0.0.1:8050/')
        except Exception:
            pass

        audio_thread = threading.Thread(target=play_audio, args=(track,))
        audio_thread.start()
        audio_thread.join()


if __name__ == "__main__":
    main()