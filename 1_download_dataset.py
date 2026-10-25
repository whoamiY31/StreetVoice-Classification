"""
Amac: 
    ESC-50 veri setini indirip , zip dosyasını aç ve proje klasörü içerisinde kullanıma hazır hale getir.

Veri seti:
    1-ESC-50 , çevresel ses sınıflandırma için kullanılan bir veriseti
    2- 2000 tane .wav formatında ses var
    3- Her ses yaklaşık 5 saniye 
    4-50 farklı sınıf var , her sınıftan 40 örnek var.

Adımlar : 
    1- Gerekli kütüphanelerin içeriye aktarılması
    2- Veri seti baglantısının ve klasor yollarının tanımlanması
    3- Dosya indirme fonksiyonu hazırlanması
    4- Zip dosyasını cıkarma fonksiyonu hazırlanması
    5- Ana calısma akısının olusturulması
    6- Veri setinin indirilmesi 
    7- Zip dosyasının cıkarılması
    8- Sonuc yollarının ekrana yazdırılması    
"""
#1- Gerekli kütüphanelerin içeriye aktarılması
import os
import zipfile
import requests 
from tqdm import tqdm

#2- Veri seti baglantısının ve klasor yollarının tanımlanması

DATASET_URL="https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
RAW_DATA_DIR="data/raw" # ham veri dosyası
ZIP_DATA=os.path.join(RAW_DATA_DIR,"esc50.zip") # zip dosya yolu 
#3- Dosya indirme fonksiyonu hazırlanması
def download_file(url,save_path):

    response=requests.get(url,stream=True) # dosyayı parca parca indirmek icin istek gönderiyoruz
    response.raise_for_status() # indirme isteginde hata varsa programı durdurur
    total_size=int(response.headers.get("content-length",0)) # toplam boyutu byte cinsi alma
    with open(save_path,"wb") as file:
        with tqdm(
            total_size,
            unit="B",
            unit_scale=True,
            desc="Veri seti indiriliyor"
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024): # dosyayı 1024 bytle parcalar halinde indirir 
                if chunk:
                    file.write(chunk) #indirilen parcayı dosyaya yazar
                    progress_bar.update(len(chunk))

#4- Zip dosyasını cıkarma fonksiyonu hazırlanması

def extract_zip(zip_path,extract_to):
    """
    zip dosyasını belirlenen klasöre çıkart

    """
    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        zip_ref.extractall(extract_to) # zip dosyasındaki tüm içeriği hedef(extract_to) çıkar


#5- Ana calısma akısının olusturulması

def main():
    "veri setini indirme ve cıkarma islemlerini sırayla calıstırır"

    #6- Veri setinin indirilmesi
    os.makedirs(RAW_DATA_DIR,exist_ok=True) # data/raw yoksa olustur 
    if not os.path.exists(ZIP_DATA): # daha once indirilmemişse
        print("Veri seti indiriliyor")
        download_file(DATASET_URL,ZIP_DATA)
    else:
        print("Zip dosyası zaten indirildi") 
    #7- Zip dosyasının cıkarılması
    extracted_folder=os.path.join(RAW_DATA_DIR,"ESC-50-master")
    if not os.path.exists(extracted_folder):
        print("Zip acılıyor")
        extract_zip(ZIP_DATA,RAW_DATA_DIR)
    else:
        print("zaten cıkarılmıs")
    #8- Sonuc yollarının ekrana yazdırılması 
    audio_path=os.path.join(extracted_folder,"audio")
    metadata_path=os.path.join(extracted_folder,"meta","esc50.csv")
    print("işlem tamamlandı")
    print(f"Ses dosyalarının yolu {audio_path}")
    print(f"metadata dosyalarının yolu {metadata_path}")

if __name__ =="__main__" :
    main()

