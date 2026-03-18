import asyncio
import json
import logging
import os
import traceback
from typing import Optional, AsyncIterable, AsyncGenerator
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
from livekit.plugins import elevenlabs, noise_cancellation, silero, liveavatar
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
_conversation_history = []  # Track conversation with timestamps
_last_message_time = 0.0  # Track when the last message ended
_last_user_speech_time = None  # Track when user last spoke
_silence_warnings_given = 0  # Count of silence warnings


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
    
    async def transcription_node(
        self, text: AsyncIterable, model_settings
    ) -> AsyncGenerator:
        """Capture timing information from TTS-aligned transcriptions"""
        global _conversation_history, _time_elapsed
        
        collected_text = ""
        start_time_val = None
        end_time_val = None
        
        async for chunk in text:
            # Check if chunk has timing attributes (duck typing instead of isinstance)
            if hasattr(chunk, 'start_time') and hasattr(chunk, 'end_time'):
                # Track timing from timed string objects
                if start_time_val is None:
                    start_time_val = chunk.start_time
                end_time_val = chunk.end_time
                collected_text += str(chunk)
                logger.info(f"📝 Timed chunk: '{chunk}' ({chunk.start_time:.2f}s - {chunk.end_time:.2f}s)")
            else:
                # Regular string chunk
                collected_text += str(chunk)
            
            yield chunk
        
        # After collecting the full message, add to history
        if collected_text and start_time_val is not None and end_time_val is not None:
            # Calculate absolute time since interview start
            elapsed_start = (_time_elapsed if _time_elapsed else 0) + start_time_val
            elapsed_end = (_time_elapsed if _time_elapsed else 0) + end_time_val
            
            _conversation_history.append({
                "role": "assistant",
                "text": collected_text.strip(),
                "start_time": elapsed_start,
                "end_time": elapsed_end,
            })
            logger.info(f"📊 Tracked agent message: {collected_text[:50]}... ({elapsed_start:.1f}s - {elapsed_end:.1f}s)")
    
    def _build_instructions(self, config: dict) -> str:
        """Build dynamic system prompt based on interview configuration"""
        
        # Example transcript for tone/style
        example_transcript = """
EXAMPLE INTERVIEW (match this professional, direct tone - don't need to follow exactly just an idea of a vibe):

Officer: Hello. Please state your name for the record.
Applicant: My name is Anand Gur.
Officer: Thank you, Anand. I'm going to ask you a few questions regarding your application to study in the United States. Please tell me — why do you want to study in the United States?
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
        
        # Get interview language
        interview_language = config.get('interviewLanguage', 'en')
        
        # Language mapping for human-readable names
        language_names = {
            'en': 'English',
            'es': 'Spanish (Español)',
            'fr': 'French (Français)',
            'hi': 'Hindi (हिंदी)',
            'ar': 'Arabic (العربية)',
            'zh': 'Chinese (中文)',
            'pt': 'Portuguese (Português)',
            'de': 'German (Deutsch)',
            'ja': 'Japanese (日本語)',
            'ko': 'Korean (한국어)',
        }
        
        language_name = language_names.get(interview_language, 'English')
        
        # Check for dual participant interview (marriage/fiance visa)
        is_dual_participant = config.get('isDualParticipant', False)
        participant1_name = config.get('participant1Name', '')
        participant2_name = config.get('participant2Name', '')
        
        # Debug logging for dual participant
        logger.info(f"👥 DUAL PARTICIPANT CHECK:")
        logger.info(f"   isDualParticipant: {is_dual_participant}")
        logger.info(f"   participant1Name: '{participant1_name}'")
        logger.info(f"   participant2Name: '{participant2_name}'")
        
        # Build participant context
        participant_context = ""
        if is_dual_participant and participant1_name and participant2_name:
            logger.info(f"✅ BUILDING DUAL PARTICIPANT CONTEXT for {participant1_name} and {participant2_name}")
            participant_context = f"""
DUAL PARTICIPANT INTERVIEW:
This is a marriage/fiancé visa interview with TWO participants present:
- {participant1_name} (U.S. Citizen Petitioner)
- {participant2_name} (Foreign National Beneficiary)

CRITICAL INSTRUCTIONS FOR DUAL INTERVIEWS:
1. Address participants by name when directing questions
2. You can ask questions to EITHER participant
3. Direct relationship questions to both: "Tell me, {participant1_name}, how did you two meet?"
4. Ask verification questions to each separately: "{participant2_name}, when did you first visit the United States?"
5. Use context clues to determine who is responding:
   - If they mention being the U.S. citizen → {participant1_name}
   - If they mention being from another country → {participant2_name}
   - If unclear, you can ask: "And which one of you is answering?"
6. Test consistency: Ask similar questions to both and compare answers
7. Assess relationship authenticity by asking both partners about shared experiences

EXAMPLE DUAL INTERVIEW FLOW:
Officer: "Good afternoon. {participant1_name} and {participant2_name}, thank you for coming today."
Officer: "{participant1_name}, tell me, how did you two meet?"
[{participant1_name} answers]
Officer: "I see. {participant2_name}, can you tell me your version of how you met?"
[{participant2_name} answers]
Officer: "{participant2_name}, when did you first visit the United States?"
[{participant2_name} answers]

Remember: Both participants are present. You can direct questions to either one by name.
"""
        else:
            if is_dual_participant or participant1_name or participant2_name:
                logger.warning(f"⚠️ DUAL PARTICIPANT DATA INCOMPLETE:")
                logger.warning(f"   isDualParticipant={is_dual_participant}, name1='{participant1_name}', name2='{participant2_name}'")
        
        # Base personality
        base_instructions = f"""You are a U.S. visa officer conducting a visa interview at an embassy or consulate.

{participant_context}

LANGUAGE: Conduct this entire interview in {language_name}. Speak ONLY in {language_name}. Do not switch to English unless the applicant cannot understand {language_name}.

TONE & STYLE:
- Professional and courteous but businesslike
- Direct and efficient with questions
- Use phrases like "Very good," "I see," "Tell me..." (in {language_name})
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
        
        # Add document context if available
        document_context = config.get('documentContext', '')
        doc_context_text = ""
        if document_context:
            doc_context_text = f"""
{document_context}

IMPORTANT: Use this document information to:
- Ask informed follow-up questions
- Verify consistency between what they say and what's in their documents
- Reference specific details from their documents naturally
- Note any discrepancies or concerns
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
        
        # Add duration and depth awareness
        duration = config.get('durationMinutes', 20)
        depth = config.get('depth', 'moderate')  # 'surface', 'moderate', 'comprehensive'
        
        # Depth-specific instructions
        depth_instructions = {
            'surface': """
INTERVIEW LEVEL: BASIC (Surface-Level)
- Ask only high-level, essential questions
- Cover key topics quickly and efficiently
- Don't probe deeply unless answer raises immediate red flag
- Focus on: identity, purpose of visit, basic eligibility
- Aim for 3-5 questions per major topic area
- Target duration: ~5 minutes
""",
            'moderate': """
INTERVIEW LEVEL: STANDARD (Surface + Selective Deep Dive)
- Start with surface-level questions across all key topics
- Then choose 1-2 areas that need deeper exploration based on:
  * User's responses that seem unclear or inconsistent
  * Critical areas for this visa type (e.g., financial for F-1, ties for B-2)
- Deep dive means: Ask 5-8 follow-up questions in those 1-2 areas
- Other areas: Keep at surface level (2-3 questions)
- Target duration: ~10 minutes
""",
            'comprehensive': """
INTERVIEW LEVEL: IN-DEPTH (Comprehensive)
- Thoroughly explore ALL major sections of the question bank
- For EACH major topic area, ask:
  * Initial surface questions (2-3)
  * Follow-up probing questions (4-6)
  * Verification questions if answers are vague
- Cover: Purpose, Financial, Academic/Work, Ties, Intent to Return, Documentation
- This is the most rigorous preparation - leave no stone unturned
- Target duration: ~15 minutes
"""
        }
        
        depth_text = depth_instructions.get(depth, depth_instructions['moderate'])
        
        duration_text = f"""
{depth_text}

TIME MANAGEMENT:
- Target interview length: ~{duration} minutes
- When you receive time updates showing 80% elapsed, start wrapping up
- Real visa interviews are brief (3-7 minutes) but thorough
- Prioritize depth over breadth based on interview level above
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

{visa_context}{doc_context_text}{focus_text}
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
    global _agent_config, _session_instance, _room_context, _start_time, _time_elapsed, _conversation_history, _last_message_time, _last_user_speech_time, _silence_warnings_given
    
    # Initialize silence tracking
    _last_user_speech_time = None
    _silence_warnings_given = 0
    
    # Store room context for tool access
    _room_context = ctx
    
    # Register session end callback to capture transcript
    async def send_session_report():
        """Send session report to Next.js API when session ends"""
        
        try:
            logger.info("📊 Session ended, generating session report...")
            
            # Extract conversation history from session.history.items (LiveKit Agent SDK API)
            conversation_items = []
            
            if hasattr(session, 'history') and hasattr(session.history, 'items'):
                logger.info(f"✅ Found session.history.items")
                logger.info(f"  Total items count: {len(session.history.items)}")
                
                for idx, item in enumerate(session.history.items):
                    # Only process message items (skip function_call, function_call_output, agent_handoff)
                    if item.type == "message":
                        role = item.role  # "user" or "assistant"
                        content = item.text_content  # The actual text
                        
                        # Try to extract timing if available
                        start_time = 0
                        end_time = 0
                        
                        # Check for common timing attributes
                        if hasattr(item, 'start_time'):
                            start_time = item.start_time
                        if hasattr(item, 'end_time'):
                            end_time = item.end_time
                        if hasattr(item, 'timestamp'):
                            start_time = end_time = item.timestamp
                        if hasattr(item, 'created_at'):
                            start_time = end_time = item.created_at
                        
                        # Log all available attributes on first item to see what's available
                        if idx == 0:
                            logger.info(f"  📊 Item attributes: {[a for a in dir(item) if not a.startswith('_')][:30]}")
                        
                        conversation_items.append({
                            "type": "message",
                            "role": role,
                            "content": [{"text": content}],
                            "start_time": start_time,
                            "end_time": end_time,
                        })
                        
                        logger.info(f"  [{idx}] {role}: {content[:60]}... (t={start_time}-{end_time})")
                        
                        if item.interrupted:
                            logger.info(f"       (interrupted)")
                    
                    elif item.type == "function_call":
                        logger.info(f"  [{idx}] function_call: {item.name}")
                    
                    elif item.type == "function_call_output":
                        logger.info(f"  [{idx}] function_output: {item.name}")
                    
                    elif item.type == "agent_handoff":
                        logger.info(f"  [{idx}] agent_handoff")
                
                logger.info(f"✅ Extracted {len(conversation_items)} conversation messages")
            
            else:
                logger.warning("⚠️ Session does not have history.items")
                logger.warning(f"  Has history: {hasattr(session, 'history')}")
                if hasattr(session, 'history'):
                    logger.warning(f"  Has items: {hasattr(session.history, 'items')}")
            
            # Build session report
            session_report = {
                "room_name": ctx.room.name,
                "history": {
                    "items": conversation_items
                },
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"📊 Session report built with {len(conversation_items)} conversation items")
            
            # Hardcoded API URL for now
            next_api_url = "https://interview-app-indol.vercel.app"
            
            # Extract interview ID from room name
            room_name = ctx.room.name
            
            # Get the Next.js API URL from environment
            endpoint = f"{next_api_url}/api/interviews/session-report"
            
            logger.info(f"📤 Sending session report to: {endpoint}")
            logger.info("=" * 80)
            
            # Try to extract interview ID from room name (format: interview_user_xxx_yyy)
            interview_id = None
            if "_" in room_name:
                parts = room_name.split("_")
                if len(parts) >= 4:
                    interview_id = parts[-1]
            
            # Build expected S3 URL for recording
            # Format: https://{bucket}.s3.{region}.amazonaws.com/interviews/{interviewId}.mp4
            expected_recording_url = None
            if interview_id:
                # Get S3 bucket and region from environment
                s3_bucket = os.getenv("AWS_S3_BUCKET", "vysa-interview-recordings")
                s3_region = os.getenv("AWS_S3_REGION", "us-east-1")
                expected_recording_url = f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/interviews/{interview_id}.mp4"
                logger.info(f"📹 Expected recording URL: {expected_recording_url}")
            
            # Build payload
            payload = {
                "roomName": room_name,
                "sessionReport": session_report,
                "recordingInfo": {
                    "interviewId": interview_id,
                    "expectedRecordingUrl": expected_recording_url,
                    "note": "Recording may take 1-2 minutes to process and upload to S3"
                } if interview_id else None,
            }
            
            logger.info(f"📤 Payload size: {len(json.dumps(payload))} bytes")
            logger.info(f"📤 Payload structure: roomName={room_name}, history_items={len(session_report.get('history', {}).get('items', []))}")
            
            # Send the session report to Next.js API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json=payload
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
    
    # Reset conversation history for this session
    _conversation_history = []
    _last_message_time = 0.0
    logger.info("🔄 Conversation tracking initialized")
    
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
    
    # Get interview language from config (default to English)
    interview_language = _agent_config.get('interviewLanguage', 'en')
    logger.info(f"🌍 Interview language: {interview_language}")
    
    # Use Cartesia TTS as primary for lower latency and better reliability
    logger.info("Configuring Cartesia TTS (primary for low latency)...")
    
    # Use custom voice for English, fallback to default multilingual voice for other languages
    if interview_language == 'en':
        # Custom English voice
        tts_instance = inference.TTS(
            model="cartesia/sonic-3",
            voice="bd9120b6-7761-47a6-a446-77ca49132781",
            language="en",
        )
        logger.info("✅ Cartesia TTS configured with custom English voice")
    else:
        # Default multilingual voice with proper language setting
        tts_instance = inference.TTS(
            model="cartesia/sonic-3",
            language=interview_language,  # Set the language explicitly
        )
        logger.info(f"✅ Cartesia TTS configured with default voice for language: {interview_language}")
    
    # Create session with TTS-aligned transcripts for timing
    # Use Deepgram Nova-3 for multilingual STT support
    # AssemblyAI universal-streaming only supports English
    if interview_language == 'en':
        stt_model = "assemblyai/universal-streaming"
        logger.info(f"🎤 STT model: {stt_model} (English)")
    else:
        # Deepgram Nova-3:multi supports language switching and handles English proper nouns correctly
        stt_model = "deepgram/nova-3:multi"
        logger.info(f"🎤 STT model: {stt_model} (multilingual with language switching)")
    
    session = AgentSession(
        stt=stt_model,
        llm="openai/gpt-4.1",
        tts=tts_instance,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        use_tts_aligned_transcript=True,  # Enable timing information
    )

    _session_instance = session
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    # Track the report task so we can await it before shutdown
    report_task = None
    
    # Register session close callback to send transcript to API
    @session.on("close")
    def _on_session_close(ev):
        """Called when the agent session closes - send transcript to Next.js API"""
        nonlocal report_task
        logger.info("📊 Session close event triggered")
        logger.info(f"📊 Close reason: {ev.reason if hasattr(ev, 'reason') else 'unknown'}")
        # Create task and store reference so we can await it later
        report_task = asyncio.create_task(send_session_report())
    
    # Ensure the report task completes before shutdown
    async def await_report_task():
        nonlocal report_task
        if report_task:
            logger.info("⏳ Waiting for session report to complete...")
            await report_task
            logger.info("✅ Session report task completed")
    
    ctx.add_shutdown_callback(await_report_task)
    
    # Word corrections for common STT mishears
    WORD_CORRECTIONS = {
        # Add common mishears here - map inappropriate words to likely intended words
        # Example: If STT consistently mishears "condo" as something inappropriate
        # "inappropriate_word": "condo",
    }
    
    def sanitize_transcription(text: str) -> str:
        """Filter and correct commonly misheard words"""
        # Apply word corrections (case-insensitive)
        corrected_text = text
        for bad_word, good_word in WORD_CORRECTIONS.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(bad_word), re.IGNORECASE)
            corrected_text = pattern.sub(good_word, corrected_text)
        
        return corrected_text
    
    # Track conversation items (user speech) as they're added to history
    @session.on("conversation_item_added")
    def _on_conversation_item_added(item):
        """Track user speech with timing information"""
        global _conversation_history, _time_elapsed, _last_user_speech_time, _silence_warnings_given
        
        try:
            # Only track user messages (not agent messages - those are tracked in transcription_node)
            if hasattr(item, 'role') and item.role == "user":
                text = ""
                if hasattr(item, 'content'):
                    if isinstance(item.content, list):
                        for block in item.content:
                            if hasattr(block, 'text'):
                                text += block.text + " "
                            elif isinstance(block, str):
                                text += block + " "
                    elif isinstance(item.content, str):
                        text = item.content
                
                text = text.strip()
                
                # Apply word filtering/correction
                if text:
                    text = sanitize_transcription(text)
                
                if text:
                    # Update last user speech time
                    import time
                    _last_user_speech_time = time.time()
                    _silence_warnings_given = 0  # Reset warning count when user speaks
                    
                    # Use current elapsed time as timestamp
                    # (user speech happens "now" in the conversation)
                    _conversation_history.append({
                        "role": "user",
                        "text": text,
                        "start_time": _time_elapsed,
                        "end_time": _time_elapsed,  # Will be updated if we get duration info
                    })
                    logger.info(f"📊 Tracked user message: {text[:50]}... ({_time_elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"❌ Error tracking conversation item: {e}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)
    
    # Background task to monitor for silence/no user response
    silence_monitor_task = None
    
    async def monitor_silence():
        """Monitor for prolonged silence and prompt user or end interview"""
        global _last_user_speech_time, _silence_warnings_given
        import time
        
        SILENCE_THRESHOLD = 30  # 30 seconds of silence before warning
        WARNING_WAIT_TIME = 10  # Wait 10 seconds after warning
        MAX_WARNINGS = 2  # After 2 warnings, end interview
        
        try:
            # Wait a bit before starting to monitor (give user time to start speaking)
            await asyncio.sleep(15)
            
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Skip if user hasn't started speaking yet (still in initial greeting)
                if _last_user_speech_time is None:
                    continue
                
                current_time = time.time()
                time_since_last_speech = current_time - _last_user_speech_time
                
                # If silence detected for too long
                if time_since_last_speech > SILENCE_THRESHOLD:
                    if _silence_warnings_given < MAX_WARNINGS:
                        _silence_warnings_given += 1
                        logger.warning(f"⚠️ Silence detected for {time_since_last_speech:.0f}s - giving warning #{_silence_warnings_given}")
                        
                        # Have the agent prompt the user
                        session.generate_reply(
                            instructions="The applicant has been silent for a while. Ask if they can hear you: 'I'm sorry, I cannot hear you clearly. Can you hear me? Please respond if you're still there.'"
                        )
                        
                        # Wait for response
                        await asyncio.sleep(WARNING_WAIT_TIME)
                        
                    else:
                        # Max warnings reached, end interview
                        logger.error(f"❌ No user response after {MAX_WARNINGS} warnings - ending interview due to technical difficulties")
                        
                        session.generate_reply(
                            instructions="Tell the applicant there appears to be a technical issue and you cannot hear them, so you must end the interview now."
                        )
                        
                        # Wait for agent to finish speaking
                        await asyncio.sleep(5)
                        
                        # End the interview
                        await ctx.room.disconnect()
                        break
                        
        except asyncio.CancelledError:
            logger.info("Silence monitor task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in silence monitor: {e}")
    
    # Start silence monitoring task
    silence_monitor_task = asyncio.create_task(monitor_silence())
    
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
                
                duration_minutes = _agent_config.get('durationMinutes', 20)
                duration = duration_minutes * 60  # convert to seconds
                percentage = (elapsed / duration) * 100 if duration > 0 else 0
                
                logger.info(f"Time update: {elapsed}s elapsed ({percentage:.0f}% of {duration}s / {duration_minutes} min)")
                
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
    
    # Initialize and start LiveAvatar BEFORE starting the session (critical order!)
    logger.info("🎭 Initializing LiveAvatar...")
    avatar_id = os.getenv("LIVEAVATAR_AVATAR_ID")
    logger.info(f"  - Avatar ID: {avatar_id[:20]}..." if avatar_id else "  - Avatar ID: NOT SET")
    
    try:
        # Configure avatar
        # Note: LiveAvatar API doesn't expose video quality settings via Python SDK
        avatar = liveavatar.AvatarSession(
            avatar_id=avatar_id,
        )
        logger.info("  - AvatarSession object created")
        
        logger.info("  - Starting avatar (BEFORE session.start per docs)...")
        logger.info(f"  - Session type: {type(session)}")
        logger.info(f"  - Room type: {type(ctx.room)}")
        logger.info(f"  - Room state: {ctx.room.connection_state}")
        
        # Start avatar FIRST (per LiveAvatar docs)
        await avatar.start(session, room=ctx.room)
        logger.info("✅ LiveAvatar avatar.start() completed")
        
        # Wait a moment for LiveAvatar to publish tracks
        await asyncio.sleep(1)
        logger.info("  - Waited 1s for LiveAvatar to join and publish tracks")
        
        logger.info("✅ LiveAvatar initialized and started successfully")
        
    except Exception as e:
        logger.error(f"❌ ERROR initializing LiveAvatar: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise
    
    # NOW start the agent session (AFTER avatar is ready)
    logger.info("🚀 Starting agent session (AFTER avatar is ready)...")
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
    logger.info("✅ Connected to room successfully")
    
    # Wait a moment for participants to join
    logger.info("⏳ Waiting for user participant to join...")
    try:
        await ctx.wait_for_participant()
        logger.info("✅ User participant joined")
    except Exception as e:
        logger.warning(f"⚠️ wait_for_participant timeout or error: {e}")
        logger.warning("⚠️ Continuing anyway...")
    
    # Check room state before generating greeting
    logger.info(f"🔍 Room state before greeting:")
    logger.info(f"  - Room name: {ctx.room.name}")
    logger.info(f"  - Local participant: {ctx.room.local_participant.identity if ctx.room.local_participant else 'None'}")
    logger.info(f"  - Remote participants: {len(ctx.room.remote_participants)}")
    logger.info(f"  - Connection state: {ctx.room.connection_state}")
    
    # Re-check tracks after connection
    logger.info(f"🔍 Tracks after connection:")
    logger.info(f"  - Local tracks count: {len(ctx.room.local_participant.track_publications)}")
    for track_sid, publication in ctx.room.local_participant.track_publications.items():
        logger.info(f"    - Track: {publication.kind} | source: {publication.source} | sid: {track_sid}")
    
    # Generate initial greeting
    visa_code = _agent_config.get('visaCode', 'visa')
    logger.info(f"🎤 Generating initial greeting for {visa_code} interview...")
    try:
        session.generate_reply(
            instructions=f"Start the interview by saying 'Hello. Please state your name for the record.' Wait for their response, then acknowledge and ask your first question from the question bank about their {visa_code} visa application."
        )
        logger.info("✅ Greeting generation initiated successfully")
    except Exception as e:
        logger.error(f"❌ Error generating greeting: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    ))
