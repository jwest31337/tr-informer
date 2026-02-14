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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

WATCH_DIR = os.getenv("WATCH_DIR")
if not WATCH_DIR:
    raise ValueError("WATCH_DIR not set in .env")
WATCH_DIR = str(Path(WATCH_DIR).resolve())  # Normalize path

print(f"Starting tr-informer")
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
    print(f"Discord bot logged in as {bot.user}")

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
# File watcher & processing
# ────────────────────────────────────────────────
class CallHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in ['.wav', '.mp3']:
            return

        # Give JSON a moment to appear (sometimes written after audio)
        time.sleep(2)

        json_path = path.with_suffix('.json')
        if not json_path.exists():
            print(f"No JSON found for {path}")
            return

        print(f"New call detected: {path} + {json_path}")

        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            return

        timestamp = datetime.fromtimestamp(metadata.get('start_time', 0)).isoformat()
        talkgroup = metadata.get('talkgroup_tag', metadata.get('talkgroup', 'Unknown'))
        call_id = metadata.get('call_id', 'unknown')

        # Transcribe
        try:
            segments, info = whisper_model.transcribe(
                str(path),
                beam_size=5,
                language="en",
                vad_filter=True,
                initial_prompt=(
                    "police radio dispatch fire ems ambulance 10-4 10- codes "
                    "affirmative negative"
                )
            )
            transcript = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception as e:
            transcript = f"[Transcription failed: {str(e)}]"
            print(f"Transcription error: {e}")

        # Interpret with Ollama
        interp_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "Interpret police/fire/EMS radio transcript: summarize clearly, expand 10-codes, identify locations/incidents, note urgency."
            },
            {"role": "user", "content": transcript}
        ])
        interpretation = interp_response['message']['content']

        # Retrieve recent/similar history
        query_embedding = embedding_model.encode(transcript + " " + interpretation).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=6)
        history_context = "\n".join(
            f"{meta.get('timestamp', '?')}: {meta.get('interpretation', meta.get('transcript', ''))}"
            for meta in results['metadatas'][0] if meta
        )

        # Generate insights
        insights_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "Analyze new radio event against recent history. Identify patterns, ongoing incidents in the area."
            },
            {
                "role": "user",
                "content": f"History:\n{history_context}\n\nNew event ({talkgroup}, {timestamp}):\n{interpretation}"
            }
        ])
        insights = insights_response['message']['content']

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
            asyncio.create_task(channel.send(msg[:2000]))

# ────────────────────────────────────────────────
# Run bot + watcher
# ────────────────────────────────────────────────
async def run_watcher():
    event_handler = CallHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    observer.start()
    print(f"Started recursive file watcher on {WATCH_DIR}")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

async def main():
    # Start Discord bot in background
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    
    # Start file watcher
    await run_watcher()

if __name__ == "__main__":
    asyncio.run(main())
