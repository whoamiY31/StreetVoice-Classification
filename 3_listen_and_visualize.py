"""
Amaç:
    1.Esc-50 den bir ses dosyasını alacağız 
    2. Dalga formunu alcaz
    3. melspectogram çıkaracaz
Adımlar:
    1.Gerekli libraries aktarılması
    2.Veri seti klasor yollarının tanımlanması
    3.metadata dosyasının okunması
    4. Ornek ses dosyasının secilmesi
    5.Ses dosyasısın yuklenmesi
    6.ses bilgilerinin ekrana yazdırılması
    7.waveform grafiginin olusturulması
    8.Melspectogram hesabı yapılması
    9.Melspectogram grafiği olusturulması
    10.Ana calısma akısı ayarlanması
    
"""
#1.
import os
import pandas as pd 
import numpy as np
import librosa # ses dosyaları yüklemek ve ozelliklerini cıkarmak
import librosa.display  # ses verilerini grafik üzerinde gostermek icin kullanılıyor
import matplotlib.pyplot as plt

#2. 
DATASET_DIR="data/raw/ESC-50-master"
META_DATA=os.path.join(DATASET_DIR,"meta","esc50.csv") # etiket dosyasının yolu 
AUDIO_DIR=os.path.join(DATASET_DIR,"audio")
OUTPUT_DIR="outputs"

#3. 
def load_metadata():
    #ESC-50 metadata dosyasını okur ve dataframe olarak dondurur
    return pd.read_csv(META_DATA)

#4.

def select_audio(df,index=0):
    #veri setinden belirtilen indexe göre ornek bir ses sececez
    sample=df.iloc[index]
    filename=sample["filename"]
    category=sample["category"]
    audio_path=os.path.join(AUDIO_DIR,filename) # tam yol
    return audio_path,filename,category
#5.
def load_audio(audio_path):
    #ses dosyasını librosa ile yukler
    audio,sr=librosa.load(audio_path,sr=None) # ses dosyasini orijinal Sr ile yükler
    return audio,sr
#6.
def show_audio_info(audio,sr,filename,category,audio_path):
    duration=len(audio) / sr # ses uzunlugunu verir - sn cinsinden-
    print(f"Ornek ses dosyası bilgisi:")
    print(f"Dosya adi : {filename}")
    print(f"Sınıf {category}")
    print(f"Dosya yolu {audio_path}")
    print(f"Ornekleme orani {sr}")
    print(f"Ses uzunlugu {round(duration,1)}")
    print(f"Ses veri boyutu : {audio.shape}")

#7.
def plot_waveform(audio,sr,category):
    #ses sinyalinin waveform grafigi olusturulur ve kayıt edilir.
    os.makedirs(OUTPUT_DIR,exist_ok=True) # output klasoru yoksa olustur
    plt.figure()
    librosa.display.waveshow(audio,sr=sr)
    plt.title(f"Waveform - {category}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    save_path=os.path.join(OUTPUT_DIR,"waveform-plot.png")
    plt.savefig(save_path)
    plt.show()

# 8.
def create_mel_spectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Mel değerlerini desibele çevir
    return mel_spec_db

# 9.
def plot_mel_spec(mel_spec_db, sr, category):
    # Mel spec grafiği oluşturur ve kaydeder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {category}")
    plt.tight_layout()
    
    # Dosyayı kaydet ve figürü bellekten temizle
    save_path = os.path.join(OUTPUT_DIR, f"mel_{category}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
#10.

def main():
    df=load_metadata()
    audio_path,filename,category= select_audio(df,index=1973)

    audio,sr=load_audio(audio_path) 
    show_audio_info(audio,sr,filename,category,audio_path)

    plot_waveform(audio,sr,category) # waveform plot olusturulması
    mel_spec_db=create_mel_spectrogram(audio,sr) # mel spec deger hesapla
    plot_mel_spec(mel_spec_db,sr,category)


if __name__=="__main__":
    main()