import os
import librosa
import torch
import logging
import speech_recognition as sr
from transformers import pipeline
from transformers import logging as hf_logging
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# --- HIDE ANNOYING LOAD WARNINGS ---
hf_logging.set_verbosity_error()

# --- 1. MAP EMOTIONS TO 3 CATEGORIES ---
TEXT_EMOTION_MAP = {
    'admiration': 'Positive', 'amusement': 'Positive', 'approval': 'Positive', 
    'caring': 'Positive', 'desire': 'Positive', 'excitement': 'Positive', 
    'gratitude': 'Positive', 'joy': 'Positive', 'love': 'Positive', 
    'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive', 'surprise': 'Positive',
    'anger': 'Negative', 'annoyance': 'Negative', 'disappointment': 'Negative', 
    'disapproval': 'Negative', 'disgust': 'Negative', 'embarrassment': 'Negative', 
    'fear': 'Negative', 'grief': 'Negative', 'nervousness': 'Negative', 
    'remorse': 'Negative', 'sadness': 'Negative', 'confusion': 'Negative',
    'neutral': 'Neutral', 'curiosity': 'Neutral', 'realization': 'Neutral'
}

# --- 2. SET UP LOCAL FOLDER PATHS ---
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

text_model_path = os.path.join(project_root, 'local_model')
audio_model_path = os.path.join(project_root, 'local_audio_model')

# --- 3. LOAD LOCAL TEXT AI ---
print(f"Loading local Text AI...")
text_classifier = pipeline('text-classification', model=text_model_path, tokenizer=text_model_path)

def analyze_text_emotion(text):
    if not text.strip():
        return "Neutral", "none"
    result = text_classifier(text)[0]
    raw_emotion = result['label'].lower()
    mapped_emotion = TEXT_EMOTION_MAP.get(raw_emotion, 'Neutral')
    return mapped_emotion, raw_emotion

# --- 4. LOAD LOCAL TONE AI ---
print(f"Loading local Tone AI...")
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import numpy as np

# Load the exact models explicitly (Bypasses the FFmpeg pipeline error entirely)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_path)
tone_model = AutoModelForAudioClassification.from_pretrained(audio_model_path)

def analyze_tone_emotion(audio_file_path):
    print(f"Analyzing tone for: {audio_file_path}")
    
    # 1. Read the audio safely using librosa (NO FFMPEG!)
    speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
    
    # 2. Process audio for the AI
    inputs = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # 3. Get the prediction
    with torch.no_grad():
        logits = tone_model(**inputs).logits
        
    # Convert math to percentages
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probabilities, 3)
    
    print("   [Tone AI Breakdown]:")
    for i in range(3):
        emo_name = tone_model.config.id2label[top_indices[i].item()].lower()
        score = top_probs[i].item() * 100
        print(f"      - {emo_name}: {score:.1f}%")
        
    # 4. Get the highest prediction
    predicted_id = top_indices[0].item()
    raw_emotion = tone_model.config.id2label[predicted_id].lower()
    
    # 5. Map the labels
    if raw_emotion in ['angry', 'sad', 'fear', 'disgust']:
        mapped_emotion = "Negative"
    elif raw_emotion in ['happy', 'pleasant_surprise']:
        mapped_emotion = "Positive"
    else:
        mapped_emotion = "Neutral"
        
    return mapped_emotion, raw_emotion

# --- 5. SPEECH TO TEXT ---
recognizer = sr.Recognizer()

def transcribe_audio(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "[Unintelligible Audio]"
        except sr.RequestError as e:
            return f"[API Error: {e}]"

# --- 6. THE INTEGRATION LOGIC MATRIX ---
def get_final_outcome(text_emotion, tone_emotion):
    if text_emotion == 'Positive' and tone_emotion == 'Positive': return 'Positive'
    if text_emotion == 'Positive' and tone_emotion == 'Neutral': return 'Positive'
    if text_emotion == 'Neutral' and tone_emotion == 'Positive': return 'Positive'
    
    if text_emotion == 'Neutral' and tone_emotion == 'Neutral': return 'Neutral'
    if text_emotion == 'Positive' and tone_emotion == 'Negative': return 'Neutral'
    if text_emotion == 'Negative' and tone_emotion == 'Positive': return 'Neutral'
    
    if text_emotion == 'Negative' and tone_emotion == 'Negative': return 'Negative'
    if text_emotion == 'Negative' and tone_emotion == 'Neutral': return 'Negative'
    if text_emotion == 'Neutral' and tone_emotion == 'Negative': return 'Negative'
    
    return 'Neutral' 

# --- 7. MAIN EXECUTION PIPELINE ---
def process_customer_audio(audio_file_path):
    print(f"\n" + "="*50)
    print(f"PROCESSING NEW CUSTOMER AUDIO")
    print(f"="*50)
    
    text = transcribe_audio(audio_file_path)
    print(f"[Speech-to-Text] : '{text}'\n")
    
    text_mapped, text_raw = analyze_text_emotion(text)
    print(f"[Text AI] Raw Output : {text_raw}")
    print(f"[Text AI] Category   : {text_mapped}\n")
    
    tone_mapped, tone_raw = analyze_tone_emotion(audio_file_path)
    print(f"[Tone AI] Raw Output : {tone_raw}")
    print(f"[Tone AI] Category   : {tone_mapped}\n")
    
    final_emotion = get_final_outcome(text_mapped, tone_mapped)
    print(f"*** FINAL MATRIX RESULT: {final_emotion.upper()} ***")
    print(f"="*50 + "\n")
    
    return final_emotion

if __name__ == "__main__":
    sample_audio = os.path.join(project_root, 'data', 'test.wav')
    if os.path.exists(sample_audio):
        final_result = process_customer_audio(sample_audio)
    else:
        print(f"Please place a 'test.wav' file in: {sample_audio}")