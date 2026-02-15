import os
import json
import time
from datetime import datetime
from pathlib import Path
import asyncio
import subprocess

import discord
from discord.ext import commands
from dotenv import load_dotenv
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from collections import deque

# ────────────────────────────────────────────────
# Load environment variables
# ────────────────────────────────────────────────
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not set in .env")

DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
if DISCORD_CHANNEL_ID == 0:
    raise ValueError("DISCORD_CHANNEL_ID not set or invalid in .env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "cogito:latest")

WATCH_DIR = os.getenv("WATCH_DIR")
if not WATCH_DIR:
    raise ValueError("WATCH_DIR not set in .env")
WATCH_DIR = str(Path(WATCH_DIR).resolve())

BATCH_INTERVAL = int(os.getenv("BATCH_INTERVAL", "900"))  # 15 minutes default

print(f"Starting tr-informer - Direct NFS polling mode")
print(f" - Watching directory: {WATCH_DIR}")
print(f" - Ollama model: {OLLAMA_MODEL}")
print(f" - Discord channel ID: {DISCORD_CHANNEL_ID}")
print(f" - Batch summary every {BATCH_INTERVAL} seconds")

# ────────────────────────────────────────────────
# Initialize components
# ────────────────────────────────────────────────
whisper_model = WhisperModel(
    "medium.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=4
)
print("Whisper model loaded.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

chroma_client = chromadb.PersistentClient(path="./tr_informer_db")
collection = chroma_client.get_or_create_collection(name="radio_history")

# Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"*** DISCORD BOT LOGGED IN AS {bot.user} ***")
    print(f" - Connected to {len(bot.guilds)} servers")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    print(f" - Target channel: {channel.name if channel else 'NOT FOUND'} (ID: {DISCORD_CHANNEL_ID})")

@bot.command(name="query")
async def query_history(ctx, *, question: str):
    results = collection.query(query_texts=[question], n_results=8)
    history_snippets = [
        meta.get('interpretation', meta.get('transcript', ''))
        for meta in results['metadatas'][0] if meta
    ]
    history_context = "\n\n".join(history_snippets)[:4000]

    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are a radio dispatch analyst. Use history to answer concisely."},
        {"role": "user", "content": f"History:\n{history_context}\n\nQuestion: {question}"}
    ])
    await ctx.send(response['message']['content'][:2000])

# ────────────────────────────────────────────────
# Batch buffer & summary
# ────────────────────────────────────────────────
call_buffer = deque(maxlen=100)  # Keep last 100 calls for batching
last_batch_time = time.time()

async def batch_summarize_and_post():
    global last_batch_time
    if time.time() - last_batch_time < BATCH_INTERVAL or len(call_buffer) == 0:
        return

    print(f"Creating batch summary for {len(call_buffer)} calls")

    batch_text = ""
    for call in call_buffer:
        batch_text += f"{call['timestamp']} {call['talkgroup']}: {call['transcript'][:200]}...\n"

    try:
        summary_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "You are a radio dispatch monitor. Summarize recent calls concisely: key events, locations, units, ongoing incidents. Use timeline format if multiple calls."
            },
            {"role": "user", "content": f"Recent calls:\n{batch_text}\n\nSummarize what happened in this period."}
        ])
        batch_summary = summary_response['message']['content']
    except Exception as e:
        batch_summary = f"[Batch summary failed: {str(e)}]"
        print(f"Batch Ollama error: {e}")

    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        msg = (
            f"**Batch Update – {len(call_buffer)} calls**\n"
            f"Time window: {call_buffer[0]['timestamp']} to {call_buffer[-1]['timestamp']}\n\n"
            f"{batch_summary}"
        )
        try:
            await channel.send(msg[:2000])
            print("Batch summary posted to Discord")
        except Exception as e:
            print(f"Failed to post batch summary: {e}")

    call_buffer.clear()
    last_batch_time = time.time()

# ────────────────────────────────────────────────
# Process a single call
# ────────────────────────────────────────────────
async def process_call(audio_path: Path):
    json_path = audio_path.with_suffix('.json')
    if not json_path.exists():
        print(f"No JSON found for {audio_path}")
        return

    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return

    timestamp = datetime.fromtimestamp(metadata.get('start_time', 0)).isoformat()
    talkgroup = metadata.get('talkgroup_tag', metadata.get('talkgroup', 'Unknown'))
    call_id = metadata.get('call_id', 'unknown')

    print(f"Processing call {call_id} ({talkgroup}, {timestamp})")

    # Convert m4a to 16kHz WAV if needed
    audio_file = str(audio_path)
    if audio_path.suffix.lower() in ['.m4a', '.mp4']:
        wav_path = audio_path.with_suffix('.wav')
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            audio_file = str(wav_path)
            print(f"Converted to WAV: {wav_path}")
        except Exception as e:
            print(f"ffmpeg conversion failed: {e}")
            audio_file = str(audio_path)  # fallback

    # Transcribe
    try:
        segments, info = whisper_model.transcribe(
            audio_file,
            beam_size=7,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=(
                "Police fire EMS dispatch radio traffic, 10-codes, unit numbers, locations in the detected area, "
                "affirmative negative enroute on scene 10-4 10-50 10-20 clear copy responding priority"
            ),
            condition_on_previous_text=True
        )
        transcript = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        print(f"Transcript (first 100 chars): {transcript[:100]}...")
    except Exception as e:
        transcript = f"[Transcription failed: {str(e)}]"
        print(f"Transcription error: {e}")

    # Interpret with Ollama
    try:
        interp_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "Interpret police/fire/EMS radio transcript: summarize clearly, expand 10-codes, identify locations/incidents, note urgency."
            },
            {"role": "user", "content": transcript}
        ])
        interpretation = interp_response['message']['content']
        print(f"Interpretation (first 100 chars): {interpretation[:100]}...")
    except Exception as e:
        interpretation = f"[Ollama interpretation failed: {str(e)}]"
        print(f"Ollama interpret error: {e}")

    # History & insights
    try:
        query_embedding = embedding_model.encode(transcript + " " + interpretation).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=6)
        history_context = "\n".join(
            f"{meta.get('timestamp', '?')}: {meta.get('interpretation', '')}"
            for meta in results['metadatas'][0] if meta
        )

        insights_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "Analyze new radio event against recent history. Identify patterns, ongoing incidents in the observed area."
            },
            {
                "role": "user",
                "content": f"History:\n{history_context}\n\nNew event ({talkgroup}, {timestamp}):\n{interpretation}"
            }
        ])
        insights = insights_response['message']['content']
        print(f"Insights (first 100 chars): {insights[:100]}...")
    except Exception as e:
        insights = f"[Insights failed: {str(e)}]"
        print(f"Ollama insights error: {e}")

    # Store in ChromaDB
    doc_id = f"call_{call_id}_{timestamp.replace(':', '-')}"
    collection.add(
        ids=[doc_id],
        embeddings=[query_embedding],
        metadatas=[{
            "timestamp": timestamp,
            "talkgroup": talkgroup,
            "transcript": transcript,
            "interpretation": interpretation,
            "insights": insights
        }]
    )

    # Add to batch buffer
    call_buffer.append({
        "timestamp": timestamp,
        "talkgroup": talkgroup,
        "transcript": transcript,
        "interpretation": interpretation,
        "insights": insights
    })

    # Try batch summary
    await batch_summarize_and_post()

# ────────────────────────────────────────────────
# Polling watcher loop
# ────────────────────────────────────────────────
async def run_watcher():
    print(f"Starting polling watcher on {WATCH_DIR} (every 15 seconds)")
    known_paths = set()

    while True:
        try:
            for root, dirs, files in os.walk(WATCH_DIR):
                for filename in files:
                    if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                        audio_path = Path(root) / filename
                        str_path = str(audio_path)

                        if str_path in known_paths:
                            continue

                        known_paths.add(str_path)

                        json_path = audio_path.with_suffix('.json')

                        if json_path.exists():
                            print(f"Poll detected new call: {audio_path} + {json_path}")
                            await process_call(audio_path)
                        else:
                            print(f"Poll found audio but no JSON yet: {audio_path}")
        except Exception as e:
            print(f"Polling loop error: {e}")

        await asyncio.sleep(15)

# ────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────
async def main():
    print("Attempting Discord login...")
    bot_task = asyncio.create_task(bot.start(DISCORD_TOKEN))

    try:
        await asyncio.wait_for(bot_task, timeout=60)
        print("Bot login task completed")
    except asyncio.TimeoutError:
        print("Discord login timed out (60s) - check token/network")
    except Exception as e:
        print(f"Discord login failed: {type(e).__name__}: {e}")

    print("Starting polling watcher...")
    await run_watcher()

if __name__ == "__main__":
    asyncio.run(main())
