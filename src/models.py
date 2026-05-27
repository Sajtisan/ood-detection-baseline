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
    Komplex (Residual) architektúra közvetlenül implementálva (segédfüggvény nélkül).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Kezdeti konvolúció a bemeneti képeken
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # ==========================================
    # 1. RESIDUAL BLOKK (32 filter, nincs méretcsökkentés)
    # ==========================================
    shortcut_1 = x
    
    # Fő ág
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Összeadás és aktiváció
    x = layers.Add()([shortcut_1, x])
    x = layers.Activation('relu')(x)

    # ==========================================
    # 2. RESIDUAL BLOKK (64 filter, méretcsökkentés: stride=2)
    # ==========================================
    # Mivel változik a csatornák száma (32 -> 64) és a térbeli méret (stride=2), 
    # a shortcut ágon is végre kell hajtani egy 1x1-es konvolúciót, hogy összeadhatóak legyenek.
    shortcut_2 = layers.Conv2D(64, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')(x)
    shortcut_2 = layers.BatchNormalization()(shortcut_2)

    # Fő ág
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Összeadás és aktiváció
    x = layers.Add()([shortcut_2, x])
    x = layers.Activation('relu')(x)

    # ==========================================
    # 3. RESIDUAL BLOKK (128 filter, méretcsökkentés: stride=2)
    # ==========================================
    shortcut_3 = layers.Conv2D(128, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')(x)
    shortcut_3 = layers.BatchNormalization()(shortcut_3)

    # Fő ág
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Összeadás és aktiváció
    x = layers.Add()([shortcut_3, x])
    x = layers.Activation('relu')(x)

    # ==========================================
    # PREDICKIÓS FEJ (Kimenet)
    # ==========================================
    # Paraméterszám optimalizálása Flatten helyett a komplex hálózatnál is
    x = layers.GlobalAveragePooling2D()(x)
    
    # SZIGORÚAN lineáris aktiváció a logitok kinyeréséhez
    outputs = layers.Dense(num_classes, activation='linear', name='logits_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_Complex')
    return model

if __name__ == "__main__":
    print("⏳ MLP Baseline modell generálása (MNIST dimenziók)...")
    mlp_model = build_mlp_baseline(input_shape=(28, 28, 1), num_classes=10)
    print("\n✅ MLP Baseline architektúra sikeresen felépítve!")
    print("📋 MLP Modell összefoglaló (Summary):\n")
    mlp_model.summary()
    print("⏳ Standard CNN modell generálása (CIFAR-10 dimenziók)...")
    try:
        cnn_model = build_cnn_standard(input_shape=(32, 32, 3), num_classes=10)
        if cnn_model is not None:
            print("\n✅ Standard CNN architektúra sikeresen felépítve!")
            print("📋 CNN Modell összefoglaló (Summary):\n")
            cnn_model.summary()
        else:
            print("❌ A CNN modell még nincs implementálva (None értéket adott vissza).")
    except Exception as e:
        print(f"❌ Hiba a CNN modell felépítése során: {e}")
     
    print("\n--------------------------------------------------\n")
     
    print("⏳ Complex ResNet modell generálása (CIFAR-10 dimenziók)...")
    try:
        resnet_model = build_resnet_complex(input_shape=(32, 32, 3), num_classes=10)
        if resnet_model is not None:
            print("\n✅ Complex ResNet architektúra sikeresen felépítve!")
            print("📋 ResNet Modell összefoglaló (Summary):\n")
            resnet_model.summary()
        else:
            print("❌ A ResNet modell még nincs implementálva (None értéket adott vissza).")
    except Exception as e:
        print(f"❌ Hiba a ResNet modell felépítése során: {e}")
