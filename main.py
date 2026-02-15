import os
import json
import time
from datetime import datetime
from pathlib import Path
import asyncio

import discord
from discord.ext import commands
from dotenv import load_dotenv
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

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

print(f"Starting tr-informer - Direct NFS polling mode")
print(f" - Watching directory: {WATCH_DIR}")
print(f" - Ollama model: {OLLAMA_MODEL}")
print(f" - Discord channel ID: {DISCORD_CHANNEL_ID}")

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
    try:
        channel = bot.get_channel(DISCORD_CHANNEL_ID)
        print(f" - Target channel: {channel.name if channel else 'NOT FOUND'} (ID: {DISCORD_CHANNEL_ID})")
    except Exception as e:
        print(f" - Channel lookup failed: {e}")

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

    # Transcribe
    try:
        segments, info = whisper_model.transcribe(
            str(audio_path),
            beam_size=5,
            language="en",
            vad_filter=True,
            initial_prompt=(
                "police radio dispatch fire ems ambulance 10-4 10- codes "
                "affirmative negative"
            )
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

    # Retrieve history & generate insights
    try:
        query_embedding = embedding_model.encode(transcript + " " + interpretation).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=6)
        history_context = "\n".join(
            f"{meta.get('timestamp', '?')}: {meta.get('interpretation', meta.get('transcript', ''))}"
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

    # Post to Discord
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        msg = (
            f"**New Call: {talkgroup} – {timestamp}**\n"
            f"**Transcript:** {transcript[:300]}{'...' if len(transcript) > 300 else ''}\n"
            f"**Interpretation:** {interpretation[:400]}{'...' if len(interpretation) > 400 else ''}\n"
            f"**Insights:** {insights[:500]}{'...' if len(insights) > 500 else ''}"
        )
        try:
            await channel.send(msg[:2000])
            print(f"Successfully posted to Discord channel {DISCORD_CHANNEL_ID}")
        except Exception as e:
            print(f"Failed to send Discord message: {e}")
    else:
        print(f"Channel {DISCORD_CHANNEL_ID} not found or bot not in server")

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
                    if filename.lower().endswith(('.wav', '.mp3', '.m4a')):  # added .m4a from your ls
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

    # Wait up to 60 seconds for login
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
