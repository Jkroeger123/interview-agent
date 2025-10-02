import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import elevenlabs, noise_cancellation, silero, tavus
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)  # Enable debug logging

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a U.S. visa officer conducting a visa interview at an embassy or consulate. 
            You are professional but firm, occasionally impatient, and somewhat skeptical of the applicant's responses.
            Your job is to determine if the applicant is eligible for a U.S. visa by asking probing questions about their:
            - Purpose of visit to the United States
            - Ties to their home country (job, family, property)
            - Financial situation and ability to support themselves
            - Travel history
            - Immigration intent (whether they plan to return home)
            
            Your tone is businesslike and occasionally annoyed, especially if answers are vague or unconvincing.
            You interrupt if answers are too long, and you're skeptical of rehearsed responses.
            Keep your questions direct and don't be overly friendly. You're busy and have many applicants to process.
            Your responses are brief, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            Speak naturally as if conducting a real interview, with occasional sighs or expressions of impatience.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents.llm import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, ElevenLabs, AssemblyAI, and the LiveKit turn detector
    logger.info("Initializing ElevenLabs TTS with voice_id=39RWTTKuyH2ra0eFxGkf")
    
    try:
        tts_instance = elevenlabs.TTS(
            voice_id="39RWTTKuyH2ra0eFxGkf",  # Your custom ElevenLabs voice
            model="eleven_multilingual_v2"
        )
        logger.info("ElevenLabs TTS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
        raise
    
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # Using ElevenLabs plugin to support custom voice IDs
        tts=tts_instance,
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Add Tavus virtual avatar to the session
    # Get your API key from https://platform.tavus.io/
    import os
    avatar = tavus.AvatarSession(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),  # Your Tavus replica ID
        persona_id=os.getenv("TAVUS_PERSONA_ID"),  # Your Tavus persona ID
    )
    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()
    
    # Generate initial greeting to start the interview immediately
    logger.info("Generating initial visa interview greeting")
    session.generate_reply(
        instructions="Greet the visa applicant briefly and ask them to state their purpose for wanting to visit the United States. Be direct and slightly impatient, as you have many applicants to process today."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
