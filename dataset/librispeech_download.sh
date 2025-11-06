set -e
mkdir -p datasets/librispeech
cd datasets/librispeech
echo "Downloading LibriSpeech dataset..."
wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c https://www.openslr.org/resources/12/train-other-500.tar.gz
wget -c https://www.openslr.org/resources/12/test-clean.tar.gz
wget -c https://www.openslr.org/resources/12/test-other.tar.gz
echo "Extracting..."
for f in *.tar.gz; do
    echo "➡️ Extracting $f ..."
    tar -xzf "$f"
done
echo "✅ LibriSpeech download + extraction completed"
