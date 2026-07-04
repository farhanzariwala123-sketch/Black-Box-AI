import torchaudio
from speechbrain.inference.interfaces import foreign_class

# 1. Load the pre-trained HuggingFace Model (Downloads locally on first run)
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

def analyze_tone(audio_file_path):
    # 2. Run the audio file through the model
    out_prob, score, index, text_lab = classifier.classify_file(audio_file_path)
    raw_emotion = text_lab[0].lower() # e.g., 'ang', 'sad', 'hap', 'neu'
    
    # 3. Map to your 3 required categories
    if raw_emotion in ['ang', 'sad']:
        return "Negative"
    elif raw_emotion in ['hap', 'exc']:
        return "Positive"
    else:
        return "Neutral"

# Example Usage:
# result = analyze_tone("customer_audio_sample.wav")
# print(f"Customer Tone: {result}")