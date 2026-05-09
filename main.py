from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from walkie_sdk import WalkieRobot

ZENOH_PORT = 7447

ROBOT_IP = "127.0.0.1"

def get_robot():
    return WalkieRobot(
        ip=ROBOT_IP,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )

def main():
    robot = get_robot()
    walkieAI = WalkieAIClient()
    walkie = WalkieInterface(robot)
    audio_stream = walkieAI.tts.synthesize_stream("Hello My name is walkie! Do you know about EIC?")
    walkie.speaker.play_stream(audio_stream, blocking=True)
    # print("Audio stream generated")
    # speaker.play_stream(audio_stream)
    # print("Audio stream played")

    # Streaming audio from the microphone
    # print(list_audio_devices())
    # microphone = Microphone()
    # print("Microphone initialized")
    # audio_stream = microphone.record_until_silence()
    # print("Audio stream recorded")
    # text = walkieAI.stt.transcribe(audio_stream)
    # print("Text transcribed")
    # print(text)


if __name__ == "__main__":
    main()
