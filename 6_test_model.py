"""
Amaç : 
    Eğitilen CNN modeline test verisi üzerinde degerlendir ve başarı metriklleri ile incele 
Veri Seti : Test veri setini kullan 

Adımlar : 
    1.Gerekli kütüphanelerin import edilmesi
    2.Veri,model ve çıktılarımızı nereye kayıt edileceğini göster.
    3.Test ayarlarının tanımlanması 
    4.Sınıf isimlerinin dosyadan okunması
    5. Test veri setini olustur 
    6.Egitilmis modelin yüklenmesi 
    7.Test verisi üzerinde tahmin yapılması - PREDICTION
    8.Başarı metriklerinin hesaplanması 
    9.Conf Matrix grafiginin olusturulması
    10.Hepsinin birlestirilmesi - ana calısma akisinin olusturulmasi
"""

#1.Gerekli kütüphanelerin import edilmesi
import os 
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#2.Veri,model ve çıktılarımızı nereye kayıt edileceğini göster.
DATASET_DIR = "data/processed/melspectograms" 
TEST_DIR=os.path.join(DATASET_DIR,"test")
MODEL_PATH="models/best_esc50_cnn_model.keras" # best model path 
CLASS_NAME_PATH="models/class_names.txt"
OUTPUT_DIR="outputs"

#3.Test ayarlarının tanımlanması 
IMG_SIZE=(128,128) # modele verilecek görsellerin boyutu 
BATCH_SIZE=32 # modelin aynı anda isleyebilecegi ornek sayısı 

#4.Sınıf isimlerinin dosyadan okunması

def load_class_names():
    """egitim sırasında kaydedilen sınıf isimlerinin dosyadan okunması"""
    with open(CLASS_NAME_PATH,"r",encoding="utf-8") as file:
        class_names=[line.strip() for line in file.readlines()]
    return class_names

#5. Test veri setini olustur 
def create_test_dataset():
    """test gorsellerini tensorflow veriseti olarak hazırlar"""

    test_ds=tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    return test_ds


#6.Egitilmis modelin yüklenmesi 

def load_trained_model():
    """kayıt edilmiş modelin yüklenmesi"""
    model=tf.keras.models.load_model(MODEL_PATH) # egitilmis modeli .keras dosyasından yükler
    return model 


#7.Test verisi üzerinde tahmin yapılması - PREDICTION

def predict_test_data(model,test_dataset):
    """test veriseti üzerinden tahmin yapar ve gercek-tahmin etiketleri dondurur"""
    y_true=[] # gercek sınıf index tutar
    y_pred=[] # tahmin edilen sınıf index tutar
    y_confidence = [] # tahmin güven skorları


    for images,labels in test_dataset: # test veri setindeki batchleri sırasıyla gezer
        predictions=model.predict(images,verbose=0) # modelin sınıf olasıklarını üretmesini saglar 
        true_classes=np.argmax(labels.numpy(),axis=1) # 
        
        predicted_classes=np.argmax(predictions,axis=1) # en yuksek olasılıklı sınıfı tahmin olarak alır 
        confidence_scores = np.max(predictions,axis=1) # her tahmin icin en yüksek olasılık degerini alir 

        y_true.extend(true_classes) # gercek sınıf indekslerini listeye ekler
        y_pred.extend(predicted_classes) # tahmin edilen sınıf indekslerini listeye ekler
        y_confidence.extend(confidence_scores) # güven skorlarını listeye ekler

    return y_true,y_pred,y_confidence


#8.Başarı metriklerinin hesaplanması 


def calculate_metrics(y_true,y_pred,class_names):
    """accuracy and class report sonuclarını hesaplar"""
    accuracy=accuracy_score(y_true,y_pred)
    print(f"Accuracy \n{accuracy}")

    clf_report=classification_report(y_true,y_pred,target_names=class_names)
    print(f"Precission,recall,f1 score and accuracy \n{clf_report}")


#9.Conf Matrix grafiginin olusturulması


def save_cnfmatrix(y_true,y_pred,class_names):
    """conf matrix olusturur"""
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    cm=confusion_matrix(y_true,y_pred) # gercek ve tahmin sınıflarına göre conf matrix olusturur 
    plt.figure()
    plt.imshow(cm,interpolation="nearest",cmap="coolwarm") 
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names,rotation=90)
    plt.yticks(tick_marks,class_names)

    plt.xlabel("Tahmin edilen sınıf")
    plt.ylabel("Gercek sınıf")

    plt.tight_layout()

    save_path=os.path.join(OUTPUT_DIR,"conf_matrix.png")
    plt.savefig(save_path)
    plt.show()

#10.Hepsinin birlestirilmesi - ana calısma akisinin olusturulmasi


def main():
    """Model test adimlarini sirasiyla calistirir"""
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    if not os.path.exists(MODEL_PATH): # model dosyasinin olup olmadigini kontrol eder 
        print("Model dosyasi bulunamadi")
        return 

    if not os.path.exists(TEST_DIR): # test dosyasinin olup olmadigini kontrol eder 
        print("Test dosyasi bulunamadi")
        return 

    #sinif isimlerinin cagirilmasi
    class_names=load_class_names() 
    #test verisetini olusturur
    test_ds=create_test_dataset()
    #model yükler
    model=load_trained_model()

    y_true,y_pred,y_confidence=predict_test_data(model,test_ds) # test veri üzerinden tahmin eder
    calculate_metrics(y_true,y_pred,class_names) # test başarı metriklerini hesaplar 

    save_cnfmatrix(y_true,y_pred,class_names) # conf matrix olustur ve kaydet 

if __name__ == "__main__":
    main()