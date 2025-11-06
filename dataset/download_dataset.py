import os
import gdown
import tarfile

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Output folder
target_folder = "datasets/librispeech"
os.makedirs(target_folder, exist_ok=True)

files = {
    "1ErM3XrVCnCJY9w2xQq5wEFc-RRM1gygr": "train-clean-100.tar.gz",
    "1aNN8Ec8M0nFANM7QM2GxeZCgnUu_TQTu": "train-clean-360.tar.gz",
    "103pthiZM6fTlW52wYYNn4aMFIvnwgpmV": "train-other-500.tar.gz",
    "171gClb43XR8yu8ir61HyGllYTDhal1RO": "dev-clean.tar.gz",
    "16xRJb1Jk2QkjtvyFW7FxMPLfI2vlFFLJ": "test-clean.tar.gz",
}


print("🔥 Starting dataset download...")

for file_id, filename in files.items():
    output_path = os.path.join(target_folder, filename)

    if os.path.exists(output_path):
        print(f"✅ Already downloaded: {filename}")
        continue

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️ Downloading: {filename}")
    gdown.download(url, output_path, quiet=False)

print("✅ All files downloaded.")


print("\n📦 Extracting .tar.gz files...")

for filename in os.listdir(target_folder):
    if filename.endswith(".tar.gz"):
        file_path = os.path.join(target_folder, filename)
        print(f"➡️ Extracting: {filename}")

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(target_folder)

print("\n🎉 DONE! LibriSpeech dataset is ready.")
print(f"📁 Saved in: {os.path.abspath(target_folder)}")
