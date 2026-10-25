"""
Amaç:-Mel spectogramları kullanarak CNN tabanlı bir ses sınıflandırma modeli eğiteceğiz.

veri seti: data/processed/melspectograms/train-val

Adımlar:
    1. Gerekli kütüphanelerin içeri aktarılması
    2.Veri ve model klasör yollarının tanımlanması
    3.Egitim ayarlarının tanımlanması 
    4.Train ve Val veri setlerinin olusturulması
    5.CNN model mimarisinin hazırlanması
    6.Egitim grafiklerini kayıt eden bir fonksiyon hazırlanması
    7.Ana calısma akısı olustur
    8.Model egitimi
    9.En iyi modeli ve sınıf isimlerinin kayıt edilmesi


"""

#1.Gerekli kütüp aktarılması
import os 
import tensorflow as tf 
from tensorflow.keras import layers,models # cnn katmanları ve model yapısı icin kullanılır 
import matplotlib.pyplot as plt 

#2.Veri ve model klasör yollarının tanımlanması
DATASET_DIR = "data/processed/melspectograms" 
TRAIN_DIR=os.path.join(DATASET_DIR,"train")
VAL_DIR=os.path.join(DATASET_DIR,"val")
MODEL_DIR="models" 
OUTPUT_DIR="outputs"

#3.Egitim ayarlarının tanımlanması 
IMG_SIZE=(128,128) # modele verilecek görsellerin boyutu 
BATCH_SIZE=32 # modelin aynı anda isleyebilecegi ornek sayısı 
EPOCH=150 # Egitim tur sayısı 
LEARNING_RATE=0.0001 # ogrenme orani 

#4.Train ve Val veri setlerinin olusturulması --> modele uygun hale getirmek 

def create_dataset():
    """Train ve validasyon gorsellerini tensorflow veri seti olarak hazırla,

    train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTONE)
    .cache() --> hafızaya al , tekrar diske gitme , RAM
    .shuffle() --> Klasorden dosyaları hep aynı sırayla alma
    .prefetch(buffer_size=AUTOTONE) --> CPU bos durma GPU 32 lik foto islerken sen diger paketi getir

    """
    train_ds=tf.keras.utils.image_dataset_from_directory( # egitim gorsellerini klasor ısımlerine gore veri setine cevirir
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    val_ds=tf.keras.utils.image_dataset_from_directory(# val gorsellerini klasor ısımlerine gore veri setine cevirir
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    class_names=train_ds.class_names # sınıf isimlerini klasör adlarından alır 
    AUTOTUNE=tf.data.AUTOTUNE # veri okuma performansı 
    train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # pretech -> prefetch yapıldı
    val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE) # val karısık okumaya gerek yok 
    return train_ds , val_ds,class_names




#5.CNN model mimarisinin hazırlanması

def build_cnn_model(num_classes):
    """melspectogram sınıflandıracak CNN modelini olusturur"""
    model=models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0],IMG_SIZE[1],3)), #modelin giris goruntu boyutu -128-128-3
            layers.Rescaling(1.0/255), # goruntulerde bulunan piksel degerlerinin 0-1 arasına sıkıstırma
            # feature extraction katmanları
            layers.Conv2D(32,(3,3),activation="relu",padding="same"), # ilk temel gorsel ozellikleri cıkarır 32 filtre sayısı 3,3 boyutu
            layers.MaxPool2D((2,2)), # gorsel boyutu kucultur
            
            layers.Conv2D(64,(3,3),activation="relu",padding="same"), # daha belirgin spectogram oruntulerini ogrenir
            layers.MaxPool2D((2,2)), #ozellik haritasini kucultur

            layers.Conv2D(128,(3,3),activation="relu",padding="same"), # daha karmasık ses-gorsel ozellikleri ogrenilir
            layers.MaxPool2D((2,2)), #hesaplama yukunun azalması

            layers.Conv2D(256,(3,3),activation="relu",padding="same"), # ust seviye ozellik cıkarımı
            layers.GlobalAveragePooling2D(), # flatten yerine daha az parametreli ozet ozellik cıkarıcı
            #class katmanı
            layers.Dense(128,activation="relu"),
            layers.Dropout(0.5), # overfitting azaltma 
            layers.Dense(num_classes,activation="softmax"), # 50 sınıf icin her bir sınıfın olma olasılıgını hesaplar -- output layer
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model=build_cnn_model(50)

#6.Egitim grafiklerini kayıt eden bir fonksiyon hazırlanması

def plot_training_history(history):
    """Eğitim ve validasyon accuracy /loss grafiklerini oluşturur"""
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy") # train acc degerlerini cizdir
    plt.plot(history.history["val_accuracy"], label="Val Accuracy") # val acc degerlerini cizdir
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"training_accuracy.png"))
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss") # train acc degerlerini cizdir
    plt.plot(history.history["val_loss"], label="Val Loss") # val acc degerlerini cizdir
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"training_loss.png"))
    plt.show()

#Ana calısma akısı olustur

def main():
    """CNN model egitim adımlarını sırasıyla calıstır"""

    os.makedirs(MODEL_DIR,exist_ok=True)

    train_ds,val_ds,class_names=create_dataset()# train ve val veri setlerini olusturur
    num_classes=len(class_names) # toplam sınıf sayısını hesaplar 

    model=build_cnn_model(num_classes) # CNN modeli olustur 
    model.summary() # model mimarisini ekrana yazdırır 

    #check point path olustur 
    checkpoint_path=os.path.join(MODEL_DIR,"best_esc50_cnn_model.keras") # en iyi modelin kayıt yolu 
    checkpoint=tf.keras.callbacks.ModelCheckpoint( # val acc en iyi oldugu modeli kaydeder
        checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        verbose=1 
    )

    early_stopping=tf.keras.callbacks.EarlyStopping( # erken durdurma 7 adımda 
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    #8.Modelin Eğitilmesi 
    history=model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCH,
        callbacks=[checkpoint,early_stopping]
    )

    #9.Modelin ve sınıf isimlerinin kayıt edilmesi 
    final_model_path=os.path.join(MODEL_DIR,"final_esc50_cnn_model.keras")  # final modelin kayıt yolu 
    model.save(final_model_path)

    class_names_path=os.path.join(MODEL_DIR,"class_names.txt")
    with open(class_names_path,"w",encoding="utf-8") as file: # sınıf isimleri dosyasını yazma modunda ac
        for class_name in class_names:
            file.write(class_name +"\n")

    plot_training_history(history)

    print("Eğitim Tamamlandı ")

if __name__ == "__main__":
    main()


"""
summary path : "C:\Users\ysfgzbrn\OneDrive - İzmir Katip Çelebi Üniversitesi\Masaüstü\street_audio_classification\READ.ME.docx"
"""