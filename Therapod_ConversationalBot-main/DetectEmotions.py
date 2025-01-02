import numpy as np
import torch
import speech_recognition as sr
import keyboard  # For key press detection
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from transformers import pipeline
import time
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er", device_map="cpu")
emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er", device_map="cpu")

# Function to detect voice emotions
def detect_voice_emotions(emotion_model, emotion_processor, emotion_labels, audio_data, sample_rate=16000):
    inputs = emotion_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        emotion_logits = emotion_model(inputs.input_values).logits
    probs = F.softmax(emotion_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 3)
    return [(emotion_labels[i], top_probs[0][idx].item()) for idx, i in enumerate(top_indices[0])]

# Function to analyze text-based emotions
def analyze_text_emotions(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device_map="cpu")
    results = pipe(text, top_k=None)
    top_3_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    return [(res['label'], res['score']) for res in top_3_results]

# Function to listen for commands triggered by the spacebar
def listen_for_commands():
    emotion_labels = ["neutral", "happy", "angry", "sad", "fear", "surprise", "disgust"]

    recognizer = sr.Recognizer()
    
    # Specify the microphone index for the ReSpeaker 4 Mic Array
    mic_index = 0  # Update this with the correct device index for your system

    with sr.Microphone(device_index=mic_index) as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Press the ENTER to start listening for up to 10 seconds.")
        # final_output = None
        # while final_output is None:
        trigger = input('PRESS ENTER')
        if trigger == "":
            print("Listening for 10 seconds...")
            try:
                # Listen for 7 seconds with a longer timeout before processing (ensures we catch the first sound)
                audio = recognizer.listen(source, timeout=12, phrase_time_limit=10)  # Increased timeout to 10 seconds

                # Record the start time
                start_time = time.time()

                command_text = recognizer.recognize_google(audio)
                raw_audio_data = audio.get_raw_data()
                print("done listening.")
                # # Voice emotion detection
                # voice_emotions = detect_voice_emotions(emotion_model, emotion_processor, emotion_labels, audio_data_np)
                # voice_emotion_output = ", ".join([f"{label} ({prob:.2f})" for label, prob in voice_emotions])
                
                # # Text-based emotion analysis
                text_emotions = analyze_text_emotions(command_text)
                text_emotion_output = ", ".join([f"{label} ({score:.2f})" for label, score in text_emotions])

                  # f"The person's speech patterns seem to indicate {voice_emotion_output}, "
                # Final formatted output
                final_output = f"{command_text}. In your response, consider the sentiment behind the person's text is {text_emotion_output}."
                print("Final Output:")
                print(final_output)

                # Record the end time
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time

                # Print the elapsed time
                print(f"AQUIRING SPEECH TO TEXT ran for {elapsed_time:.2f} seconds")
                return final_output, command_text

            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Speech Recognition error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

