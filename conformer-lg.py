from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
import torch
import soundfile as sf

# Load teacher
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
teacher = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
teacher.eval()

# Freeze
for p in teacher.parameters():
    p.requires_grad = False

speech, sr = sf.read("example.wav")
input_list = [speech, speech]
inputs = processor(input_list, sampling_rate=sr, return_tensors="pt", padding=True)

print(speech.shape)
with torch.no_grad():
    logits = teacher(**inputs).logits  # shape: [batch, time, vocab_size]

# Optionally decode
predicted_ids = torch.argmax(logits, dim=-1)
transcriptions = processor.batch_decode(predicted_ids)
print(transcriptions)
