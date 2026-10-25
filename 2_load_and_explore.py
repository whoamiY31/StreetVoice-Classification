"""
Amaç : 
    ESC-50 veri setini keşif etmek 
Veri seti : 
    1. meta/esc50.csv dosyasını kullanacağız
    2.csv dosyasının icerisinde ses dosya isimleri, sınıfşaro , hedef etiketleri , fold bilgileri 

Adımlar : 
    1. Gerekli kütüphane içeriye aktarılması
    2. veri seti klasor yolları tanımlanması 
    3.metadata dosyasının okunması 
    4.veri setinin ilk satırlarının incelenmesi 
    5. veri seti boyutu ve sütunlarının incelenmesi
    6. eksik deger kontrolunun yapılması 
    7. sınıf sayısı ve sınıf dagılımı incelenmesi
    8. fold dagılım incelenmesi
    9.sınıf dagılım grafiginin olusturulması 
    10. Ornek bir ses dosyasının kontrol edilmesi     
"""
#1. 
import os
import pandas as pd
import matplotlib.pyplot as plt 
# 2. 
DATASET_DIR="data/raw/ESC-50-master"
META_DATA=os.path.join(DATASET_DIR,"meta","esc50.csv") # etiket dosyasının yolu 
AUDIO_DIR=os.path.join(DATASET_DIR,"audio")
OUTPUT_DIR="outputs"

# 3.  and 4.
def load_metadata():
    # esc50 metadata dosyasının okur ve dataframe donusturur
    df=pd.read_csv(META_DATA) # esc50.csv tablo formatı okur 
    return df


#5.
def show_dataset_info(df):
    print(f"Veri seti boyutu \n{df.shape}")
    print(df.columns.tolist())

#6.

def check_missing_values(df):
    print(f"Eksik veri kontrolü \n {df.isnull().sum()}") # her sutunda olan eksik deger sayısını gösterir

#7. 
def show_class_dist(df):
    """Sınıf sayısını ve her sınıftaki örnek sayısını gösterir."""
    print(f"Toplam sınıf sayısı: \n{df['category'].nunique()}")  # Benzersiz sınıf sayısını gösterir
    print(f"Her sınıftaki örnek sayısı: \n{df['category'].value_counts()}")

#8.
def show_fold_dist(df):
    # veri setindeki fold dagılımını gösterir 
    print(f"Fold dagilimi \n{df['fold'].value_counts().sort_index()}")
#9.
def plot_class_dist(df):
    #sınıf dagılımını bar grafik olarak olusturur ve kayıt eder
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    class_counts=df["category"].value_counts() #her siniftaki ornek sayisini hesaplar
    plt.figure()
    class_counts.plot(kind="bar")
    plt.title("ESC-50 class contribution ")
    plt.xlabel("Ses siniflari")
    plt.ylabel("Ornek sayisi")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_path=os.path.join(OUTPUT_DIR,"class_distribution.png") #kayıt dosyası yolu olustur
    plt.savefig(save_path)
    plt.show()

#10. Tek bir ses dosyasını inceleyelim 

def check_sample_audio_file(df):
    #ilk ses dosyasini klasörde bulunup bulunmadıgının kontrol eder 

    sample_file=df.iloc[0]["filename"] # ilk ses dosyasinin adini alir
    sample_category=df.iloc[0]["category"] # ilk ses dosyasinin sinif adini alır 
    sample_path=os.path.join(AUDIO_DIR,sample_file)

    if os.path.exists(sample_path):
        print("Dosya mevcut")
    else:
        print("Not exist")



df=load_metadata()
print(df.head())
show_dataset_info(df)
check_missing_values(df)
show_class_dist(df)
show_fold_dist(df)
plot_class_dist(df)
check_sample_audio_file(df)




