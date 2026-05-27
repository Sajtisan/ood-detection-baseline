import os
from data_loader import load_mnist_id, load_cifar10_id
from preprocessing import create_dataset_pipeline
from models import build_mlp_baseline, build_cnn_standard, build_resnet_complex
from train import train_model
from visualization import plot_training_history

def main():
    # 1. Adatok betöltése és előkészítése
    print("[1/2] MNIST adatok betöltése...")
    (x_train_m, y_train_m), (x_val_m, y_val_m) = load_mnist_id()
    mnist_train_ds = create_dataset_pipeline(x_train_m, y_train_m, batch_size=64, is_training=True)
    mnist_val_ds = create_dataset_pipeline(x_val_m, y_val_m, batch_size=64, is_training=False)

    print("[2/2] CIFAR-10 adatok betöltése...")
    (x_train_c, y_train_c), (x_val_c, y_val_c) = load_cifar10_id()
    cifar_train_ds = create_dataset_pipeline(x_train_c, y_train_c, batch_size=64, is_training=True)
    cifar_val_ds = create_dataset_pipeline(x_val_c, y_val_c, batch_size=64, is_training=False)

    # 2. Modellek listája
    # Egy szótárban tároljuk a modelleket és a hozzájuk tartozó adatot
    models_to_train = [
        {"name": "MLP_MNIST", "builder": build_mlp_baseline, "shape": (28, 28, 1), "train_ds": mnist_train_ds, "val_ds": mnist_val_ds, "epochs": 20},
        
        {"name": "CNN_CIFAR", "builder": build_cnn_standard, "shape": (32, 32, 3), "train_ds": cifar_train_ds, "val_ds": cifar_val_ds, "epochs": 50},
        {"name": "ResNet_CIFAR", "builder": build_resnet_complex, "shape": (32, 32, 3), "train_ds": cifar_train_ds, "val_ds": cifar_val_ds, "epochs": 50}
    ]

    # 3. Modellek tanítása
    for m in models_to_train:
        print(f"\n\nKísérlet indítása: {m['name']} ...")
        try:
            # Modell felépítése
            model = m['builder'](input_shape=m['shape'], num_classes=10)
            
            if model is None:
                print(f"A {m['name']} modell még nincs implementálva. Ugrás a következőre...")
                continue
                
            # Tanítás
            history = train_model(
                model=model,
                train_dataset=m['train_ds'],
                val_dataset=m['val_ds'],
                epochs=m['epochs'],
                save_path=f"saved_models/{m['name'].lower()}.keras"
            )
            
            # Grafikon generálása
            plot_training_history(history, model_name=m['name'])
            
        except Exception as e:
            print(f"❌ Hiba a {m['name']} modell tanítása során: {e}")
            print("Folytatás a következő modellel...\n")

    print("\n✅ MINDEN ELÉRHETŐ KÍSÉRLET LEFUTOTT!")

if __name__ == "__main__":
    main()