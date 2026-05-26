import os
from dotenv import load_dotenv

# Betölti a lokális .env fájlt
load_dotenv()

# Hardver konfigurációk kiolvasása és beállítása
# (Ezeket az operációs rendszer szintjén kell beállítani, még mielőtt a TensorFlow betöltődik!)
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = os.getenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")

# Útvonalak beállítása (alapértelmezett értékkel, ha nincs .env)
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR", "./saved_models")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

# Hiperparaméterek (Stringből int-té alakítva)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))