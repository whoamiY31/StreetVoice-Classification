"""
Amaç:
    1- ESC-50 veri setindeki seslerin mel-spektrogram dönüşümünü yapmak ve eş zamanlı data augmentation uygulamak.
"""

# 1- Gerekli kütüphaneler
import os 
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# 2- Yol tanımlamaları
DATASET_DIR = "data/raw/ESC-50-master"
META_DATA = os.path.join(DATASET_DIR, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
OUTPUT_DIR = "data/processed/melspectograms"

# 3- Split belirleme
def get_split_name(fold):
    """Fold değerine göre train, val veya test bilgisini döndürür."""
    if fold in [1, 2, 3]:
        return "train"
    elif fold == 4:
        return "val"
    elif fold == 5:
        return "test"
    else:
        raise ValueError(f"Beklenmeyen fold değeri: {fold}")

# 4- Noise augmentation
def add_noise(audio, noise_factor=0.005):
    """Sinyale düşük seviyeli Gaussian gürültü ekler."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio.astype(np.float32)

# 5- Time shift augmentation
def time_shift(audio, shift_max=0.2):
    """Sinyali zaman ekseninde rastgele kaydırır."""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    augmented_audio = np.roll(audio, shift)
    return augmented_audio.astype(np.float32)  # Düzeltildi: astpye -> astype

# 6- Pitch shift augmentation
def pitch_shift(audio, sr, n_steps=2):
    """Ses perdesini rastgele değiştirir."""
    step = np.random.uniform(-n_steps, n_steps)
    augmented_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=step)
    return augmented_audio.astype(np.float32)

# 7- Mel-spektrogram görselini kaydetme
def save_mel_spec_image(audio, sr, save_path):
    """Mel-spektrogramı hesaplar, kenar boşluksuz olarak belirtilen yola kaydeder."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, cmap="magma", ax=ax)
    
    # Düzeltildi: save_path doğrudan dışarıdan gelen yolla kaydedilir
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# 8- Bir ses dosyasından orijinal ve türetilmiş görsellerin üretilmesi
def creates_images_for_audio(audio_path, output_dir, base_filename, split_name):
    audio, sr = librosa.load(audio_path, sr=22050)
    orijinal_path = os.path.join(output_dir, f"{base_filename}_orijinal.png")

    if not os.path.exists(orijinal_path):
        save_mel_spec_image(audio, sr, orijinal_path)

    if split_name != "train":
        return

    noise_path = os.path.join(output_dir, f"{base_filename}_noise.png")
    shift_path = os.path.join(output_dir, f"{base_filename}_shift.png")
    pitch_path = os.path.join(output_dir, f"{base_filename}_pitch.png")

    # Düzeltildi: os.path.exists ve fonksiyon argümanları (sr kaldırıldı / doğru parametre verildi)
    if not os.path.exists(noise_path):
        noise_audio = add_noise(audio)
        save_mel_spec_image(noise_audio, sr, noise_path)
        
    if not os.path.exists(shift_path):
        shift_audio = time_shift(audio)
        save_mel_spec_image(shift_audio, sr, shift_path)
        
    if not os.path.exists(pitch_path):
        pitch_audio = pitch_shift(audio, sr=sr)
        save_mel_spec_image(pitch_audio, sr, pitch_path)

# 9- Metadata okuma
def load_metadata():
    return pd.read_csv(META_DATA)

# 10- Tüm ses dosyalarını işleme
def process_audio_files(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Spektrogramlar Üretiliyor"):
        filename = row["filename"]
        category = row["category"]
        fold = row["fold"]
        split_name = get_split_name(fold)

        audio_path = os.path.join(AUDIO_DIR, filename)
        class_dir = os.path.join(OUTPUT_DIR, split_name, category)
        base_filename = os.path.splitext(filename)[0]

        os.makedirs(class_dir, exist_ok=True)

        if not os.path.exists(audio_path):
            print(f"Uyarı: Ses dosyası bulunamadı -> {audio_path}")
            continue

        creates_images_for_audio(audio_path, class_dir, base_filename, split_name)

# 11- Ana akış
def main():
    df = load_metadata()
    process_audio_files(df)
    print(f"\nİşlem tamamlandı. Çıktılar: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()