import os
import sys
import argparse
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from loguru import logger

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.frames.frames import LLMRunFrame
from pipecat.transcriptions.language import Language

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="INFO")

async def run_bot(webrtc_connection):
    """
    This function is called by the RequestHandler when a user connects.
    """
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY"), 
        model="saarika:v2.5"
    )
    
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"), 
        model="gemini-2.5-flash"
    )
    
    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY"),
        voice_id="shubh",
        model="bulbul:v3",
        params=SarvamTTSService.InputParams(language=Language.EN, pace=1.0),
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers short."},
    ]
    context = LLMContext(messages)
    
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),      # Get Audio
        stt,                    # Audio -> Text
        user_aggregator,        # Wait for silence
        llm,                    # Text -> Thinking
        tts,                    # Thinking -> Audio
        transport.output(),     # Play Audio
        assistant_aggregator,   # Save History
    ])

    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("✅ Client connected!")
        messages.append({"role": "system", "content": "Say hello."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("❌ Client disconnected")
        await task.cancel()

    # 5. Run
    runner = PipelineRunner()
    await runner.run(task)


app = FastAPI()

small_webrtc_handler = SmallWebRTCRequestHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """
    1. Browser sends Offer.
    2. We create a connection object.
    3. We start 'run_bot' in the background with that connection.
    4. We return the Answer.
    """
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection)

    return await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )

@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle network negotiation updates (ICE candidates)."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}

@app.get("/")
async def serve_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)