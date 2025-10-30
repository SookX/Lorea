# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-medium")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium")

conv1 = model.model.encoder.conv1
conv2 = model.model.encoder.conv2


print(model)
pytorch_total_params = sum(p.numel() for p in conv2.parameters())
print(f"Total model parameters: {pytorch_total_params}")