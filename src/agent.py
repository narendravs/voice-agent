import logging
import textwrap
import aiohttp
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    inference,
    function_tool,
    RunContext,
    room_io,
)
from livekit.plugins import ai_coustics

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Dynamic Environment Router for LangGraph API Endpoint
# Set LANGGRAPH_ENV="production" or "local" in your environment or .env file
ENV_MODE = os.getenv("LANGGRAPH_ENV", os.getenv("NEXT_PUBLIC_LANGGRAPH_ENV", "production")).lower()

DEFAULT_PROD_URL = "https://healthcare-eight-mu.vercel.app/api/aiagents/voiceAgent"
DEFAULT_LOCAL_URL = "http://localhost:3000/api/aiagents/voiceAgent"

if ENV_MODE == "local":
    LANGGRAPH_API_ENDPOINT = os.getenv("NEXT_PUBLIC_LOCAL_LANGGRAPH_AGENT_ENDPOINT", DEFAULT_LOCAL_URL)
    logger.info(f"LangGraph Agent configured for LOCAL environment: {LANGGRAPH_API_ENDPOINT}")
else:
    LANGGRAPH_API_ENDPOINT = os.getenv("NEXT_PUBLIC_PROD_LANGGRAPH_AGENT_ENDPOINT", DEFAULT_PROD_URL)
    logger.info(f"LangGraph Agent configured for PRODUCTION environment: {LANGGRAPH_API_ENDPOINT}")

# Define a timeout for your backend calls (e.g., 30 seconds)
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

def _session_id() -> str:
    """Use the LiveKit room name as a unique session id per call."""
    try:
        return get_job_context().room.name
    except Exception:
        return "voice-session-default"

async def call_backend(query: str) -> tuple[bool, str]:
    """POST a natural-language query to the backend.

    Returns (ok, text). Every request and response is logged so you can see
    exactly what the backend said. Failures are returned as ok=False instead of
    being replaced with made-up fallback text.
    """
    payload = {"query": query, "sessionId": _session_id(), "language": "en"}
    logger.info(f"[backend] POST {LANGGRAPH_API_ENDPOINT} payload={payload}")

    # Optional: include Vercel protection bypass header if configured
    headers = {"Content-Type": "application/json"}
    vercel_bypass = os.getenv("VERCEL_PROTECTION_BYPASS_SECRET")
    if vercel_bypass:
        headers["x-vercel-protection-bypass"] = vercel_bypass

    try:
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as http:
            async with http.post(
                LANGGRAPH_API_ENDPOINT,
                json=payload,
                headers=headers,
            ) as resp:
                raw = await resp.text()
                logger.info(f"[backend] status={resp.status} body={raw[:500]}")

                if resp.status != 200:
                    return False, f"The scheduling system returned HTTP {resp.status}."

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Plain-text or streamed response
                    return True, raw.strip()

                if isinstance(data, dict):
                    for key in ("output", "response", "message", "reply", "answer", "text"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return True, val.strip()
                    if data.get("error"):
                        return False, str(data["error"])

                return True, json.dumps(data)[:1000]
    except Exception as e:
        logger.exception(f"[backend] request failed: {e}")
        return False, "Could not reach the scheduling system."


def _failure(message: str) -> str:
    return (
        f"ERROR: {message} Tell the user briefly that you couldn't complete this right now "
        "and offer to try again. Do NOT invent doctors and do NOT say an appointment was booked."
    )

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="openai/gpt-4o-mini"),
            # To use a realtime model instead of a voice pipeline, replace the LLM
            # with a realtime model and remove the STT/TTS from the AgentSession
            # (Note: This is for OpenAI GPT-Live, the recommended speech-to-speech
            # model. For other providers, see https://docs.livekit.io/agents/models/realtime/)
            # Replace the llm argument with the model's LiveKit Inference ID:
            #    llm="openai/gpt-live-1",
            instructions=textwrap.dedent(
                """\
                You are the voice assistant for CarePulse, a hospital appointment service.
                Today is {today}.

                # Style
                - Replies are 1 to 3 short spoken sentences. Ask ONE question at a time.
                - No lists, markdown or emojis. This is a phone call.
                - The greeting has already been spoken. Do not greet again.

                # Tools (you MUST use them)
                - You have NO knowledge of which doctors exist. Never name or guess a doctor yourself.
                - If the user asks about doctors, who is available, or who they can see, IMMEDIATELY call `get_available_doctors`.
                  Do not answer from memory and do not say you will check without actually calling it.
                  Then read out the doctor names from the tool result.
                - Use doctor names exactly as returned by `get_available_doctors`.
                - If a tool result starts with "ERROR", apologize briefly, say you could not complete it, and offer to retry.

                # Booking an appointment
                Collect these four details, one at a time, in this order:
                  1. The patient's full name.
                  2. Which doctor they want to see.
                  3. Preferred date and time.
                  4. Reason for the visit or symptoms.
                - If the user already gave a detail, do not ask for it again.
                - If they pick a doctor and you have not fetched the list yet, call `get_available_doctors` first to verify the name.
                - As soon as you have all four, call `schedule_appointment` right away. Do not ask for extra confirmation.
                - Convert the date and time to the format YYYY-MM-DD HH:MM (24-hour, {TIMEZONE.key}) using today's date above.
                  Example: "tomorrow at 3 pm" becomes the next day's date followed by 15:00.
                - After the tool returns, tell the user the outcome in one or two sentences.
                - If the tool result asks for more information (for example phone number or email),
                  ask the user for it, then call `schedule_appointment` again with the same details
                  and put the new information in `additional_info`.
                """
            ),
        )

        
    # ==========================================
    # VALIDATION TOOL: Get Available Doctors
    # ==========================================
    @function_tool
    async def get_available_doctors(self, context: RunContext):
        """Fetch the real list of available doctors from the hospital database.

        Call this whenever the user asks who the doctors are, who is available,
        which doctor they can see, or before confirming a doctor name for booking.
        """
        logger.info("[tool] get_available_doctors called")
        ok, text = await call_backend("List all available doctors with their names and specialties.")
        if not ok:
            return _failure(text)
        return text

    # ==========================================
    # APPOINTMENT SCHEDULING TOOL (Triggered at the end)
    # ==========================================
    @function_tool
    async def schedule_appointment(
        self, 
        context: RunContext, 
        patient_name: str, 
        doctor_name: str, 
        appointment_time: str, 
        reason: str,
        additional_info: str = "",
    ) -> str:
        """Schedule a medical appointment for a patient ONLY AFTER you have collected their full name, the doctor's name, the appointment date/time, and the reason for the visit.

        Args:
            patient_name: The full name of the patient.
            doctor_name: Doctor's name exactly as returned by get_available_doctors.
            appointment_time: Date and time in the format YYYY-MM-DD HH:MM (24-hour).
            reason: The reason or medical symptoms for the visit.
            additional_info: Optional extra details requested by the system, such as phone or email.

        Returns:
            A confirmation string stating whether the appointment was successfully booked.
        """
        logger.info(
            f"[tool] schedule_appointment called: {patient_name} / {doctor_name} / {appointment_time} / {reason}"
        )

        session_id = context.room.name if hasattr(context, "room") else "voice-session-default"
        
        query = (
            f"Schedule an appointment for patient {patient_name} with doctor {doctor_name} "
            f"on {appointment_time} for reason: {reason}."
        )
        if additional_info:
            query += f" Additional details: {additional_info}."
        
        ok, text = await call_backend(query)
        if not ok:
            return _failure(text)
        return text
        
                
server = AgentServer()


@server.rtc_session(agent_name="voice-agent")
async def voice_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        # Keyterms bias the STT toward distinctive words it would otherwise misspell.
        # List your own names, brands, and jargon in `keyterms`. Detection additionally
        # extracts terms from the live conversation, such as a caller's name, and applies
        # them once the transcript corroborates the spelling.
        # See more at https://docs.livekit.io/agents/models/stt/keyterms/
        # stt_context_options=STTContextOptions(
        #     keyterms=["LiveKit"],
        #     keyterm_detection={"enabled": True},
        # ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
         
        turn_handling=TurnHandlingOptions(
        # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
        # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
        # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
        # AgentSession supplies the required VAD automatically.
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=inference.TurnDetector(),
        # Adaptive interruptions use the turn detector to tell a real interruption from a
        # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
        interruption={"mode": "adaptive"},
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation={"enabled": False},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=False,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        # room_options=room_io.RoomOptions(
        #     audio_input=room_io.AudioInputOptions(
        #         noise_cancellation=ai_coustics.audio_enhancement(
        #             model=ai_coustics.EnhancerModel.QUAIL_VF_S
        #         ),
        #     ),
        # ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)
    
    # Join the room and connect to the user
    await ctx.connect()

    # FIX: generate_reply() has no `text=` argument, which raised a TypeError and
    # ended the job. Use say() for an exact line.
    try:
        await session.say("Welcome to Care Pulse! How can I help you today?")
    except Exception:
        logger.exception("Greeting failed")

if __name__ == "__main__":
    cli.run_app(server)
