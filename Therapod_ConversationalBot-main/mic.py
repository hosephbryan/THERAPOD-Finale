import speech_recognition as sr

# List all available microphones
print("Available Microphones:")
mic_list = sr.Microphone.list_microphone_names()
for idx, mic in enumerate(mic_list):
    print(f"Index: {idx}, Name: {mic}")
