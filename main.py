from client import WalkieAIClient
from interfaces.speaker import Speaker
from interfaces.microphone import Microphone, list_audio_devices

def main():
    walkieAI = WalkieAIClient()
    # print("Walkie Client initialized")
    # speaker = Speaker()
    # print("Speaker initialized")
    # # Streaming audio from the microphone
    # audio_stream = walkieAI.tts.synthesize_stream("Niggers. Niggers. Niggers. Niggers.")
    # print("Audio stream generated")
    # speaker.play_stream(audio_stream)
    # print("Audio stream played")

    # Streaming audio from the microphone
    print(list_audio_devices())
    microphone = Microphone()
    print("Microphone initialized")
    audio_stream = microphone.record_until_silence()
    print("Audio stream recorded")
    text = walkieAI.stt.transcribe(audio_stream)
    print("Text transcribed")
    print(text)


if __name__ == "__main__":
    main()
