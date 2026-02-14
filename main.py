import os
import asyncio
import json
from datetime import datetime
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from pydantic import BaseModel
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import uvicorn
from faster_whisper import WhisperModel

# ────────────────────────────────────────────────
# Load environment variables
# ────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set in .env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not set in .env")

DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
if DISCORD_CHANNEL_ID == 0:
    raise ValueError("DISCORD_CHANNEL_ID not set or invalid in .env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ────────────────────────────────────────────────
# Configs & Initialization
# ────────────────────────────────────────────────
app = FastAPI(title="tr-informer")

# Faster-Whisper (CPU-optimized)
print("Loading faster-whisper model (medium.en int8 CPU)...")
whisper_model = WhisperModel(
    "medium.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    # download_root="~/.cache/whisper"
)
print("Whisper model loaded.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    # Semantic search in history
    results = collection.query(
        query_texts=[question],
        n_results=8
    )
    history_snippets = [meta.get('interpretation', meta.get('transcript', '')) 
                        for meta in results['metadatas'][0] if meta]
    history_context = "\n\n".join(history_snippets)[:4000]  # avoid token blowup

    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are a helpful radio dispatch analyst. Use provided history to answer questions concisely and factually."},
        {"role": "user", "content": f"History:\n{history_context}\n\nQuestion: {question}"}
    ])
    await ctx.send(response['message']['content'][:2000])  # Discord message limit

# API key check
def check_api_key(x_api_key: str = Form(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ────────────────────────────────────────────────
# Upload endpoint (called by trunk-recorder)
# ────────────────────────────────────────────────
@app.post("/upload")
async def handle_upload(
    call_json: str = Form(...),
    audio: UploadFile = UploadFile(...),
    api_key: str = Depends(check_api_key)
):
    # Save audio temporarily
    audio_path = f"/tmp/{audio.filename}"
    try:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # Transcribe
        try:
            segments, info = whisper_model.transcribe(
                audio_path,
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

        # Parse metadata
        try:
            metadata = json.loads(call_json)
        except:
            metadata = {}
        
        timestamp = datetime.fromtimestamp(metadata.get('start_time', 0)).isoformat()
        talkgroup = metadata.get('talkgroup_tag', metadata.get('talkgroup', 'Unknown'))
        call_id = metadata.get('call_id', 'unknown')

        # Interpret with Ollama
        interp_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "You are an expert at interpreting police/fire/EMS radio traffic. Summarize clearly, expand 10-codes, identify locations/incidents, note urgency."
            },
            {"role": "user", "content": f"Raw transcript: {transcript}"}
        ])
        interpretation = interp_response['message']['content']

        # Retrieve recent/similar history for context
        query_embedding = embedding_model.encode(transcript + " " + interpretation).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=6
        )
        history_context = "\n".join(
            f"{meta.get('timestamp', '?')}: {meta.get('interpretation', meta.get('transcript', ''))}"
            for meta in results['metadatas'][0] if meta
        )

        # Generate insights
        insights_response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": "Analyze new radio event against recent history. Identify patterns, ongoing incidents, or connections in the observed area."
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
            await channel.send(msg[:2000])  # safety cut

        return {"status": "processed", "transcript": transcript[:500]}

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

# ────────────────────────────────────────────────
# Run both FastAPI and Discord bot
# ────────────────────────────────────────────────
async def run_bot():
    await bot.start(DISCORD_TOKEN)

async def main():
    # Start Discord bot in background
    asyncio.create_task(run_bot())
    
    # Run Uvicorn (it will create/manage its own loop)
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
