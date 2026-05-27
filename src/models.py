import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp_baseline(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    
    # Rejtett rétegek he_normal inicializációval
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)

    # Kulcsfontosságú: a kimeneti réteg lineáris aktivációval, hogy megkapjuk a logitokat (nem softmax-olt értékek)
    # Erre a Temperature Scaling paramétervizsgálata miatt van szükség, hogy a logitokat közvetlenül használhassuk a hőmérséklet-skalázáshoz.
    outputs = layers.Dense(num_classes, activation='linear', name='logits_output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='MLP_Baseline')
    return model

def build_cnn_standard(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    # 1. Blok
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 2. Blok
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 3. Blok
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Paraméterszám optimalizálása Flatten helyett
    x = layers.GlobalAveragePooling2D()(x)
    
    # SZIGORÚAN lineáris aktiváció a logitok kinyeréséhez (pl. Temperature Scalinghez)
    outputs = layers.Dense(num_classes, activation='linear', name='logits_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Standard')
    return model


def build_resnet_complex(input_shape=(32, 32, 3), num_classes=10):
    """
    Komplex (Residual) architektúra (Implementációra vár).
    Cél: A hálózat mélységének és zajtűrő képességének maximalizálása.
    
    Technikai követelmények:
    - Két lehetséges megközelítés:
      1. Egyedi, egyszerűsített ResNet blokk írása (Residual connection tf.keras.layers.Add() használatával).
      2. A beépített tf.keras.applications.ResNet50V2 használata `include_top=False` beállítással, 
         majd saját predikciós fej (head) hozzáillesztése.
    - A predikciós réteg SZIGORÚAN `layers.Dense(num_classes, activation='linear')` legyen.
    """
    # TODO: A modell felépítése a fenti specifikáció alapján
    pass

if __name__ == "__main__":
    print("⏳ MLP Baseline modell generálása (MNIST dimenziók)...")
    mlp_model = build_mlp_baseline(input_shape=(28, 28, 1), num_classes=10)
    print("\n✅ MLP Baseline architektúra sikeresen felépítve!")
    print("📋 MLP Modell összefoglaló (Summary):\n")
    mlp_model.summary()

    # =====================================================================
    # TESZTEK A TOVÁBBI MODELLEKHEZ (A fejlesztés megkezdésekor vegyétek ki a kommentet)
    # =====================================================================

    # print("⏳ Standard CNN modell generálása (CIFAR-10 dimenziók)...")
    # try:
    #     cnn_model = build_cnn_standard(input_shape=(32, 32, 3), num_classes=10)
    #     if cnn_model is not None:
    #         print("\n✅ Standard CNN architektúra sikeresen felépítve!")
    #         print("📋 CNN Modell összefoglaló (Summary):\n")
    #         cnn_model.summary()
    #     else:
    #         print("❌ A CNN modell még nincs implementálva (None értéket adott vissza).")
    # except Exception as e:
    #     print(f"❌ Hiba a CNN modell felépítése során: {e}")
    # 
    # print("\n--------------------------------------------------\n")
    # 
    # print("⏳ Complex ResNet modell generálása (CIFAR-10 dimenziók)...")
    # try:
    #     resnet_model = build_resnet_complex(input_shape=(32, 32, 3), num_classes=10)
    #     if resnet_model is not None:
    #         print("\n✅ Complex ResNet architektúra sikeresen felépítve!")
    #         print("📋 ResNet Modell összefoglaló (Summary):\n")
    #         resnet_model.summary()
    #     else:
    #         print("❌ A ResNet modell még nincs implementálva (None értéket adott vissza).")
    # except Exception as e:
    #     print(f"❌ Hiba a ResNet modell felépítése során: {e}")
