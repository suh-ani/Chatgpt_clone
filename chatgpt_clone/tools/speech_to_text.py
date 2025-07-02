import speech_recognition as sr
from pydub import AudioSegment
import os

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    wav_path = audio_path
    try: 
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return "Speech recognition service error: " + str(e)
    except Exception as e:
        print("Transcription error:", e)
        return "Unexpected error during transcription."
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)