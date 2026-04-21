from client import WalkieClient
from human_interfaces.speaker import Speaker

def main():
    walkie = WalkieClient()
    print("Walkie Client initialized")
    speaker = Speaker()
    print("Speaker initialized")
    # Streaming audio from the microphone
    audio_stream = walkie.tts.synthesize_stream("Nigga")
    print("Audio stream generated")
    speaker.play_stream(audio_stream)
    print("Audio stream played")


if __name__ == "__main__":
    main()
