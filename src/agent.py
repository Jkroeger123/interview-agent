import asyncio
import json
import logging
import os
from typing import Optional
from datetime import datetime
import httpx

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents import inference
from livekit.plugins import elevenlabs, noise_cancellation, silero, tavus
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

load_dotenv(".env.local")

# Global variables for session state
_agent_config = {}
_session_instance = None
_room_context = None
_start_time = None
_time_elapsed = 0


class Assistant(Agent):
    def __init__(
        self,
        config: dict,
        ragie_global_partition: str,
    ) -> None:
        logger.info("🤖 Initializing Assistant class")
        self.ragie_global_partition = ragie_global_partition
        self.config = config
        logger.info(f"🤖 Assistant config: visa={config.get('visaCode')}")
        
        # Build dynamic instructions based on config
        logger.info("🤖 Building dynamic instructions...")
        instructions = self._build_instructions(config)
        logger.info(f"🤖 Instructions built: {len(instructions)} chars")
        
        # Initialize Agent with instructions
        # The @function_tool decorator automatically registers tools
        logger.info("🤖 Calling super().__init__ with instructions")
        super().__init__(instructions=instructions)
        logger.info("✅ Assistant initialized successfully")
    
    def _build_instructions(self, config: dict) -> str:
        """Build dynamic system prompt based on interview configuration"""
        
        # Example transcript for tone/style
        example_transcript = """
EXAMPLE INTERVIEW (match this professional, direct tone - don't need to follow exactly just an idea of a vibe):

Officer: Good morning.
Applicant: Good morning, officer.
Officer: Please give me your full name.
Applicant: My name is Anand Gur.
Officer: Nice to meet you, Anand. I'm going to ask you a few questions regarding your application to study in the United States. Please tell me — why do you want to study in the United States?
Applicant: Officer, I decided to study in the United States because of the excellent reputation of an American degree internationally. I'm a student of Hospitality and Tourism Management, and the U.S. has one of the biggest hospitality sectors in the world.
Officer: Very good. What got you interested in this field of study?
Applicant: After completing high school, I was exploring what course would suit me best. I found that hospitality was the right fit for my interests.
Officer: Very good. Have you ever studied in the United States before?
Applicant: No, sir, I haven't.
Officer: How will you fund your studies in the United States?
Applicant: My parents will be sponsoring my studies.
Officer: Do you have documentation to show that they are able to do this?
Applicant: Yes, sir, I have the documents.
Officer: Very good. Are you planning to work while you're in the United States?
Applicant: No, sir, I don't have any intention of working.
Officer: What are your plans after completing your studies?
Applicant: After completing my studies, I plan to return to my home country with the skills and knowledge I've gained. I want to establish my own business — a chain of restaurants — which will help my family and contribute to my country's economy.
Officer: That's a great goal. Finally, tell me — why do you feel that you qualify to receive a student visa today?
Applicant: Officer, I believe I'm qualified because I'm academically well-prepared, have strong English communication skills, and am financially capable. I'll follow all U.S. visa regulations and return to my country after completing my studies.
Officer: Very good, excellent. Based on your answers today, I'm happy to grant your visa to study in the United States. Congratulations!

IMPORTANT: Match this officer's tone - professional, efficient, direct. Use phrases like "Very good," ask follow-up questions naturally, and keep responses brief.
"""
        
        # Base personality
        base_instructions = f"""You are a U.S. visa officer conducting a visa interview at an embassy or consulate.

TONE & STYLE:
- Professional and courteous but businesslike
- Direct and efficient with questions
- Use phrases like "Very good," "I see," "Tell me..."
- Keep responses brief (1-2 sentences maximum)
- No emojis, asterisks, or formatting symbols
- Speak naturally as in a real interview

CRITICAL: ASK ONE QUESTION AT A TIME
- NEVER ask compound or multi-part questions
- BAD: "What school did you attend, what are you studying, and how much is tuition?"
- GOOD: "What school did you attend?" (then wait for answer, then ask next question)
- Ask follow-up questions based on their response, but ONE AT A TIME
- This is how real visa officers conduct interviews - they ask, listen, then ask again

{example_transcript}"""
        
        # Add visa-specific context (streamlined)
        visa_context = f"""
VISA TYPE: {config.get('visaCode', 'Unknown')} - {config.get('visaName', 'Unknown')}

{config.get('agentPromptContext', '')}
"""
        
        # Add focus areas if specified
        focus_areas = config.get('focusAreaLabels', [])
        focus_text = ""
        if focus_areas:
            focus_text = f"\nFOCUS AREAS: Concentrate especially on: {', '.join(focus_areas)}"
        
        # Question topics (not full questions)
        question_topics = config.get('questionTopics', [])
        question_text = ""
        if question_topics:
            question_text = f"""
QUESTION TOPICS TO COVER:
{', '.join(question_topics)}

Use the get_relevant_questions tool to fetch specific questions for any topic as needed during the interview.
"""
        
        # Add duration awareness
        duration = config.get('duration', 20)
        duration_text = f"""
INTERVIEW DURATION: {duration} minutes
- Pace yourself to cover key topics within this time
- When you receive time updates showing 80% or more of time elapsed, start wrapping up
- Real visa interviews are brief (3-7 minutes typically) and decisive
"""
        
        # Add interview strategy guidance
        doc_text = """
AVAILABLE TOOLS:

1. get_relevant_questions: Fetch specific questions for a topic (e.g., "financial", "academic", "ties to home country")
2. lookup_reference_documents: Search official visa guidelines and requirements
3. end_interview: End the session (NO PARAMETERS - you must say goodbye in conversation FIRST, then call this)

INTERVIEW STRATEGY - CRITICAL GUIDELINES:

ATOMIC QUESTIONS - ONE AT A TIME:
CRITICAL: Ask ONE question at a time. NO compound or multi-part questions.

BAD Examples:
- "What school are you attending, what will you study, and how much is tuition?"
- "Tell me about your financial sponsor and how much they earn."
- "Where did you do your undergraduate degree and what was your GPA?"

GOOD Examples:
- "What school are you attending?" (wait for answer)
- Then: "What program will you be studying?" (wait for answer)
- Then: "How much is the tuition?" (wait for answer)

WHY THIS MATTERS:
- Real visa officers ask one question at a time
- It allows you to listen and follow up naturally
- It prevents overwhelming the applicant
- It creates a more natural conversation flow

QUESTIONING APPROACH:
- Use get_relevant_questions to get main questions from the question bank
- BUT you are NOT limited to these questions - they are your foundation
- Probe deeper when answers are vague, incomplete, or raise concerns
- Ask follow-up questions naturally based on their responses
- If something doesn't make sense, dig deeper immediately
- Be conversational but maintain professional control
- REMEMBER: ONE question at a time, then WAIT for the response

FLEXIBILITY IN QUESTIONING:
- Don't just go question-by-question through the bank like a checklist
- If they mention something interesting, follow up on it before moving to the next bank question
- If an answer is weak or raises a red flag, address it immediately with a follow-up
- Skip questions if they've already been naturally answered
- Prioritize depth over breadth - better to thoroughly explore 3-4 areas than superficially cover 10

ENDING THE INTERVIEW - CRITICAL TWO-STEP PROCESS:
IMPORTANT: Ending requires TWO separate turns. DO NOT call end_interview() in the same turn as saying goodbye!

Step 1 (First Turn):
- Say your goodbye naturally: "Thank you for your time today. We'll be in touch regarding your application. Have a great day!"
- DO NOT call any tools in this turn
- Wait for the applicant to respond

Step 2 (Next Turn - AFTER they respond):
- Once they say goodbye back, THEN call end_interview()
- This ensures a natural conversation ending
"""
        
        # Combine all parts
        full_instructions = f"""{base_instructions}

{visa_context}{focus_text}
{question_text}
{duration_text}
{doc_text}
"""
        
        logger.info(f"📋 Built system instructions: {len(full_instructions)} characters")
        logger.info(f"📋 First 500 chars: {full_instructions[:500]}")
        logger.info(f"📋 Last 500 chars: {full_instructions[-500:]}")
        
        return full_instructions
    
    @function_tool
    async def get_relevant_questions(self, topic: str):
        """Fetch relevant interview questions for a specific topic.
        
        Use this to get specific questions when you want to explore a particular area of the interview.
        
        Available topics:
        - "academic" or "education": Questions about study plans, program, university
        - "financial": Questions about funding, sponsors, expenses
        - "ties" or "home country": Questions about post-graduation plans and ties to home country
        - "immigration": Questions about visa history, intent, family in U.S.
        - "english": Questions about English proficiency
        - "documents": Questions about paperwork and consistency
        - "work" or "opt": Questions about work intentions, OPT, CPT
        
        Args:
            topic: The topic area you want questions for (e.g., "financial", "academic", "ties")
        """
        logger.info(f"🔧 TOOL CALL: get_relevant_questions(topic='{topic}')")
        question_bank = self.config.get('questionBank', [])
        
        if not question_bank:
            return "No question bank available. Please ask questions based on the visa requirements."
        
        # Topic keyword mappings
        topic_keywords = {
            'academic': ['study', 'university', 'program', 'degree', 'major', 'curriculum', 'education', 'school', 'professor'],
            'financial': ['sponsor', 'fund', 'tuition', 'expense', 'income', 'bank', 'money', 'pay', 'financial', 'afford'],
            'ties': ['return', 'home country', 'after graduation', 'plans', 'ties', 'property', 'family', 'job', 'career'],
            'immigration': ['visa', 'refused', 'denied', 'overstay', 'relatives', 'Green Card', 'petition', 'immigration'],
            'english': ['English', 'TOEFL', 'IELTS', 'language', 'proficiency'],
            'documents': ['I-20', 'DS-160', 'SEVIS', 'documents', 'paperwork', 'gap', 'inconsisten'],
            'work': ['work', 'OPT', 'CPT', 'employment', 'job', 'intern', 'H-1B'],
        }
        
        # Normalize topic
        topic_lower = topic.lower()
        
        # Find matching questions
        relevant_questions = []
        keywords = []
        
        # Get keywords for this topic
        for key, words in topic_keywords.items():
            if key in topic_lower or topic_lower in key:
                keywords.extend(words)
        
        # If no specific match, use the topic itself as keyword
        if not keywords:
            keywords = [topic_lower]
        
        # Filter questions by keywords
        for q in question_bank:
            q_lower = q.lower()
            if any(keyword in q_lower for keyword in keywords):
                relevant_questions.append(q)
        
        if not relevant_questions:
            return f"No specific questions found for '{topic}'. Consider asking general questions about this area."
        
        # Return up to 10 relevant questions
        questions_to_return = relevant_questions[:10]
        formatted = "\n".join([f"- {q}" for q in questions_to_return])
        
        result = f"Relevant questions for {topic}:\n{formatted}\n\nSelect the most appropriate questions based on the conversation flow. You don't need to ask all of them."
        logger.info(f"✅ TOOL RESULT: Found {len(questions_to_return)} questions for topic '{topic}'")
        return result
    
    @function_tool
    async def lookup_reference_documents(self, question: str):
        """Look up information from official visa reference materials and guidelines.
        
        Use this when you need to verify:
        - Visa requirements and eligibility criteria
        - Legal standards and regulations
        - Common reasons for visa denial
        - Official procedures and policies
        
        This searches general reference materials, not the applicant's personal documents.
        
        Args:
            question: The specific question or topic to search reference materials for
        """
        logger.info(f"🔧 TOOL CALL: lookup_reference_documents(question='{question}')")
        
        if not self.ragie_global_partition:
            return "No reference documents partition configured."
        
        try:
            from ragie import Ragie
            
            ragie_client = Ragie(auth=os.getenv("RAGIE_API_KEY"))
            
            logger.info(f"🔍 QUERYING REFERENCE DOCUMENTS:")
            logger.info(f"   Question: {question[:100]}...")
            logger.info(f"   Partition: {self.ragie_global_partition}")
            
            results = ragie_client.retrievals.retrieve(request={
                "query": question,
                "partition": self.ragie_global_partition,
                "top_k": 3,
            })
            
            if not results or not hasattr(results, 'scored_chunks') or len(results.scored_chunks) == 0:
                logger.info("✅ TOOL RESULT: No relevant information found in reference materials")
                return "No relevant information found in reference materials."
            
            # Extract and format the relevant content
            chunks_text = []
            for chunk in results.scored_chunks[:3]:
                text = chunk.text.strip()
                chunks_text.append(text)
            
            logger.info(f"✅ TOOL RESULT: Found {len(chunks_text)} relevant chunks from reference materials")
            combined = "\n\n".join(chunks_text)
            return f"Visa regulations and requirements:\n{combined}"
            
        except Exception as e:
            logger.error(f"❌ TOOL ERROR: Error querying reference documents: {e}")
            return "Unable to access reference materials at this time."
    
    @function_tool
    async def end_interview(self):
        """End the interview session gracefully.
        
        CRITICAL: This is a TWO-TURN process!
        
        DO NOT call this tool in the same turn as your goodbye message!
        
        Turn 1: Say goodbye in normal conversation (NO tool call)
        - Example: "Thank you for your time today. We'll notify you of our decision. Have a great day!"
        - DO NOT call this tool yet
        - Wait for the applicant to respond
        
        Turn 2: AFTER they respond, THEN call this tool
        - Only call this tool after the applicant says goodbye back
        - This ensures a natural conversation ending
        
        Use this when:
        - Interview time is up (typically 5-7 minutes)
        - All necessary questions have been asked
        - You've exchanged goodbyes with the applicant
        """
        logger.info("🔧 TOOL CALL: end_interview() - Ending session and disconnecting")
        
        global _room_context
        
        # Disconnect immediately - the agent has already said goodbye in conversation
        if _room_context:
            try:
                logger.info("🔌 Disconnecting room now")
                await _room_context.room.disconnect()
                logger.info("✅ Room disconnected successfully")
                return "Interview session ended."
            except Exception as e:
                logger.error(f"❌ Error in end_interview: {e}")
                return f"Interview concluded with error: {str(e)}"
        else:
            logger.warning("⚠️ No session instance available to end interview")
            return "Unable to properly end interview - session not found"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent with session reporting enabled"""
    global _agent_config, _session_instance, _room_context, _start_time, _time_elapsed
    
    # Store room context for tool access
    _room_context = ctx
    
    # Register session end callback to capture transcript
    async def send_session_report():
        """Send session report to Next.js API when session ends"""
        try:
            logger.info("📊 Session ended, generating session report...")
            logger.info("=" * 80)
            logger.info("DEBUG: Environment and Session State")
            logger.info("=" * 80)
            
            # Hardcoded API URL for now
            next_api_url = "https://interview-app-indol.vercel.app"
            logger.info(f"🔍 API URL (hardcoded): {next_api_url}")
            
            # Get the session history directly from the session instance
            if _session_instance is None:
                logger.error("❌ Session instance is None, cannot get history")
                return
            
            logger.info(f"🔍 Session instance type: {type(_session_instance)}")
            logger.info(f"🔍 Session instance attributes: {dir(_session_instance.history)}")
            
            # Convert session history to dict
            history_dict = _session_instance.history.to_dict()
            
            logger.info(f"📊 Session report generated for room: {ctx.room.name}")
            logger.info(f"📊 Report contains {len(history_dict.get('items', []))} conversation items")
            logger.info(f"🔍 History dict keys: {history_dict.keys()}")
            
            # Log ALL items to see what's there
            all_items = history_dict.get('items', [])
            logger.info(f"🔍 Total items: {len(all_items)}")
            for idx, item in enumerate(all_items):
                logger.info(f"🔍 Item {idx}: type={item.get('type')}, role={item.get('role')}, has_content={bool(item.get('content'))}")
            
            # Log first 2 items in detail
            logger.info(f"🔍 First 2 items (detailed): {all_items[:2]}")
            
            # Build session report structure
            session_report = {
                "room_name": ctx.room.name,
                "history": history_dict,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Extract interview ID from room name
            room_name = ctx.room.name
            
            # Get the Next.js API URL from environment
            endpoint = f"{next_api_url}/api/interviews/session-report"
            
            logger.info(f"📤 Sending session report to: {endpoint}")
            logger.info("=" * 80)
            
            # Send the session report to Next.js API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json={
                        "roomName": room_name,
                        "sessionReport": session_report,
                    }
                )
                
                if response.status_code == 200:
                    logger.info("✅ Session report successfully sent to API")
                else:
                    logger.error(f"❌ Failed to send session report. Status: {response.status_code}")
                    logger.error(f"❌ Response: {response.text}")
        
        except Exception as e:
            logger.error(f"❌ Error sending session report: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Don't raise - we don't want to break the session cleanup
    
    # Add the session report callback
    ctx.add_shutdown_callback(send_session_report)
    
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # COMPREHENSIVE DEBUGGING - Let's see what's actually available
    logger.info("=" * 80)
    logger.info("DEBUGGING: Inspecting ctx and room objects")
    logger.info("=" * 80)
    
    # Log all ctx attributes
    logger.info(f"ctx attributes: {dir(ctx)}")
    logger.info(f"ctx.room type: {type(ctx.room)}")
    logger.info(f"ctx.room attributes: {dir(ctx.room)}")
    
    # Check room metadata
    logger.info(f"ctx.room.metadata type: {type(ctx.room.metadata)}")
    logger.info(f"ctx.room.metadata value: '{ctx.room.metadata}'")
    logger.info(f"ctx.room.metadata length: {len(ctx.room.metadata) if ctx.room.metadata else 0}")
    
    # Check room name
    logger.info(f"ctx.room.name: {ctx.room.name}")
    
    # Check _info (private attribute that might have metadata)
    if hasattr(ctx.room, '_info'):
        logger.info(f"ctx.room._info: {ctx.room._info}")
        if hasattr(ctx.room._info, 'metadata'):
            logger.info(f"ctx.room._info.metadata: {ctx.room._info.metadata}")
    
    # Check ctx.job for metadata
    if hasattr(ctx, 'job'):
        logger.info(f"ctx.job: {ctx.job}")
        logger.info(f"ctx.job attributes: {dir(ctx.job)}")
        if hasattr(ctx.job, 'room'):
            logger.info(f"ctx.job.room: {ctx.job.room}")
            if hasattr(ctx.job.room, 'metadata'):
                logger.info(f"ctx.job.room.metadata: {ctx.job.room.metadata}")
    
    # Check remote_participants (after connection we can check these)
    logger.info(f"ctx.room.remote_participants (should be empty before connection): {ctx.room.remote_participants}")
    
    logger.info("=" * 80)
    
    # Extract agent configuration from job room metadata
    # NOTE: ctx.room.metadata is empty at this point (before connection)
    # The metadata is available in ctx.job.room.metadata!
    _agent_config = {}
    
    if ctx.job.room.metadata:
        try:
            _agent_config = json.loads(ctx.job.room.metadata)
            logger.info(f"✅ Loaded agent config for {_agent_config.get('visaCode', 'Unknown')} visa")
            logger.info(f"✅ Question bank size: {len(_agent_config.get('questionBank', []))} questions")
            
            # Get global reference partition
            ragie_global_partition = _agent_config.get('ragieGlobalPartition', 'visa-student')
            logger.info(f"✅ Ragie global partition: {ragie_global_partition}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse room metadata: {e}")
            logger.error(f"❌ Raw metadata: {ctx.job.room.metadata[:200]}")  # First 200 chars
            ragie_global_partition = "visa-student"
    else:
        logger.warning("⚠️ No room metadata found - using default configuration")
        logger.warning(f"⚠️ Room name: {ctx.room.name}")
        ragie_global_partition = "visa-student"
    
    # Try ElevenLabs TTS first, fallback to Cartesia if it fails
    logger.info("Attempting to initialize ElevenLabs TTS...")
    tts_instance = None
    
    try:
        raise Exception("Test error")
        # Try ElevenLabs first
        tts_instance = elevenlabs.TTS(
            voice_id="39RWTTKuyH2ra0eFxGkf",  # Your custom ElevenLabs voice
            model="eleven_multilingual_v2"
        )
        logger.info("✅ ElevenLabs TTS initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ ElevenLabs TTS initialization failed: {e}")
        logger.info("Falling back to Cartesia TTS via LiveKit Inference")
        tts_instance = "cartesia/sonic-3"
        logger.info("✅ Cartesia TTS configured successfully (fallback)")
    
    # Create session
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1",
        tts=tts_instance,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    _session_instance = session
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Initialize Tavus avatar
    logger.info("Initializing Tavus avatar...")
    avatar = tavus.AvatarSession(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),
        persona_id=os.getenv("TAVUS_PERSONA_ID"),
    )
    await avatar.start(session, room=ctx.room)
    logger.info("✅ Tavus avatar initialized successfully")
    
    # Listen for time updates from frontend
    @ctx.room.on("data_received")
    def on_data_received(packet):
        global _time_elapsed, _start_time
        
        try:
            # packet is a DataPacket object, extract the data
            data = packet.data if hasattr(packet, 'data') else packet
            message = json.loads(data.decode() if hasattr(data, 'decode') else data)
            
            if message.get('type') == 'time_update':
                elapsed = message.get('elapsed', 0)
                _time_elapsed = elapsed
                
                if _start_time is None:
                    _start_time = elapsed
                
                duration = _agent_config.get('duration', 20) * 60  # convert to seconds
                percentage = (elapsed / duration) * 100 if duration > 0 else 0
                
                logger.info(f"Time update: {elapsed}s elapsed ({percentage:.0f}% of {duration}s)")
                
                # Inject wrap-up context at 80% mark
                if percentage >= 80 and percentage < 85:
                    logger.info("Interview time at 80% - should start wrapping up")
                    # Note: In the current LiveKit Agents SDK, we can't easily inject
                    # system messages mid-conversation. The agent will naturally
                    # pace itself based on the duration in the initial prompt.
            
            elif message.get('type') == 'end_interview':
                logger.info(f"🔴 Received end_interview signal from user (reason: {message.get('reason', 'unknown')})")
                logger.info("🔴 Agent leaving room to close session gracefully")
                # Disconnect the agent from the room so LiveKit can close it cleanly
                # Use asyncio.create_task since this is a sync callback
                asyncio.create_task(ctx.room.disconnect())
                
        except Exception as e:
            logger.error(f"Error processing data message: {e}")
    
    # Start the session
    logger.info("🚀 Creating Assistant instance...")
    assistant = Assistant(
        config=_agent_config,
        ragie_global_partition=ragie_global_partition,
    )
    logger.info("🚀 Starting agent session...")
    try:
        await session.start(
            agent=assistant,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("✅ Agent session started successfully")
    except Exception as e:
        logger.error(f"❌ ERROR starting agent session: {e}")
        logger.error(f"❌ Exception type: {type(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise

    # Add background "thinking" audio during tool calls
    logger.info("🎵 Initializing background thinking audio...")
    background_audio = BackgroundAudioPlayer(
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
        ],
    )
    await background_audio.start(room=ctx.room, agent_session=session)
    logger.info("✅ Background audio player started")

    await ctx.connect()
    
    # Generate initial greeting
    visa_code = _agent_config.get('visaCode', 'visa')
    logger.info(f"Generating initial greeting for {visa_code} interview")
    session.generate_reply(
        instructions=f"Greet the applicant briefly for their {visa_code} visa interview and ask your first question from the question bank. Be direct and slightly impatient, as you have many applicants to process today."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    ))
