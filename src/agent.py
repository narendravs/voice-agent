import logging
import textwrap
import aiohttp

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
                You are a helpful and friendly medical assistant voice agent.
                
                # Conversational Rules
                - Keep your replies brief: one to three sentences. Ask one question at a time.
                - When the session starts, greet the user warmly and ask how you can help them today.
                - If the user asks who the available doctors are, you MUST invoke the `get_available_doctors` tool to fetch the real-time list from the system.
                - If the user wants to schedule an appointment, collect the required details **strictly one by one**:
                  1. First, ask for the patient's full name.
                  2. Once received, ask which doctor they want to see.
                  3. Once received, ask for the preferred date and time.
                  4. Finally, ask for the reason for the visit or symptoms.
                - Do NOT call the scheduling tool until you have collected ALL four pieces of information.
                - Once all details are collected, call the `schedule_appointment` tool and summarize the result to the user.
                """
            ),
        )

    # ==========================================
    # VALIDATION TOOL: Get Available Doctors
    # ==========================================
    @function_tool
    async def get_available_doctors(self, context: RunContext):
        """Use this tool to fetch the list of available doctors from the database when the user asks who they can see.
        """
        logger.info("Fetching available doctors from backend...")
        session_id = context.room.name if hasattr(context, "room") else "voice-session-default"

        async with aiohttp.ClientSession() as session:
            try:
                # Call your Next.js API route that queries the DB for doctors
                async with session.post(
                    "http://localhost:3000/api/aiagents/voiceAgent",
                    json={"query": "Get me the list of available doctors", "sessionId": session_id}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("output", "Dr. Smith and Dr. Jones are available.")
            except Exception as e:
                logger.error(f"Failed to fetch doctors: {e}")
                
        return "We have general practitioners available."

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
        reason: str
    ):
        """Use this tool to schedule an appointment ONLY AFTER you have collected the patient name, doctor name, appointment time, and reason from the user.

        Args:
            patient_name: The full name of the patient.
            doctor_name: The name of the doctor.
            appointment_time: The requested date and time for the appointment.
            reason: The reason or medical symptoms for the visit.
        """
        logger.info(f"Triggering appointment scheduling for {patient_name} with {doctor_name}")

        session_id = context.room.name if hasattr(context, "room") else "voice-session-default"
        
        query_text = (
            f"Schedule an appointment for patient {patient_name} "
            f"with doctor {doctor_name} at {appointment_time} for reason: {reason}"
        )

        async with aiohttp.ClientSession() as session:
            try:
                # Update with your production URL when deployed
                api_url = "http://localhost:3000/api/aiagents/voiceAgent"
                
                async with session.post(
                    api_url,
                    json={
                        "query": query_text,
                        "sessionId": session_id,
                        "language": "en"
                    },
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("output", "Appointment booked successfully.")
                    else:
                        error_resp = await response.text()
                        logger.error(f"Next.js API error {response.status}: {error_resp}")
                        return "I reached the scheduling system, but encountered an error saving the appointment."
            except Exception as e:
                logger.error(f"Failed to connect to Next.js agent endpoint: {e}")
                return "Sorry, I am unable to connect to the scheduling service right now."
                
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


if __name__ == "__main__":
    cli.run_app(server)
