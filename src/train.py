import os
import tensorflow as tf
from utils import set_global_seed

def train_model(model, train_dataset, val_dataset, epochs=50, save_path="saved_models/model.keras"):
    set_global_seed(42)  # Reprodukálhatóság garantálása minden tanítási futásnál!
    # Mappa létrehozása a modell mentéséhez, ha még nem létezik
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Tanulási ráta ütemező (Cosine Decay)
    # Kiszámoljuk a teljes lépésszámot: epochok száma * batch-ek száma epochonként
    total_steps = epochs * len(train_dataset)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,  # Kezdő tanulási ráta
        decay_steps=total_steps,      # Lassan csökken a tanítás végéig
        alpha=0.01                    # A legvégén az eredeti érték 1%-ára csökken
    )

    # 3. Optimizer beállítása az ütemezővel
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 4. Veszteségfüggvény (Szigorúan from_logits=True!)
    # Mivel a models.py-ban leírtam, hogy nincs Softmax az utolsó/kimeneti rétegben,
    # ezzel a paraméterrel jelezzük a hálózatnak, hogy a kimenet nyers logit.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 5. Modell fordítása (Compile)
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn, 
        metrics=['accuracy']
    )

    # 6. Keras Callbacks (EarlyStopping és ModelCheckpoint)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor='val_loss',  # A validációs hibát figyeljük, hogy a legjobb modellt mentsük
        save_best_only=True, # Csak a legjobb modellt mentjük (a legalacsonyabb val_loss)
        save_weights_only=False, # Az egész modellt mentjük, nem csak a súlyokat
        verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # A validációs hibát figyeljük
        patience=5,          # Ha 5 egymást követő epochban nem javul a val_loss, akkor megállítjuk a tanítást
        restore_best_weights=True, # Visszaállítjuk a legjobb modell súlyait a tanítás végén
        verbose=1
    )

    print(f"\n Tanítás indítása: {model.name}")
    print(f"\n Mentési útvonal: {save_path}")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
    print(f"\n Tanítás befejezve és a legjobb modell elmentve")
    return history

if __name__ == "__main__":
    from models import build_mlp_baseline
    from data_loader import load_mnist_id
    from preprocessing import create_dataset_pipeline

    print("⏳ Rendszer tesztelése egy mini tanítási ciklussal (1 epoch)...")
    
    # 1. Adatok betöltése
    (x_train, y_train), (x_val, y_val) = load_mnist_id()
    
    # Szűkítsük le az adatot csak 1000 példára, hogy a teszt 3 másodperc alatt lefusson!
    train_dataset = create_dataset_pipeline(x_train[:1000], y_train[:1000], batch_size=32, is_training=True)
    val_dataset = create_dataset_pipeline(x_val[:200], y_val[:200], batch_size=32, is_training=False)
    
    # 2. Modell felépítése
    model = build_mlp_baseline(input_shape=(28, 28, 1), num_classes=10)
    
    # 3. Tanítás lefuttatása
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1, # Teszteléshez csak 1 epoch!
        save_path="saved_models/test_mlp_model.keras"
    )