# Variables
TRAINING_SCRIPT = train.py
MODEL_FILE = classifier.joblib
VENV_DIR = venv/

# Créer l'environnement virtuel et installer les dépendances
install:
	python3 -m venv $(VENV_DIR)
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

# Préparer les données
prepare:
	python3 $(TRAINING_SCRIPT) --prepare

# Entraîner le modèle
train:
	python3 $(TRAINING_SCRIPT) --train

# Sauvegarder le modèle
save:
	python3 $(TRAINING_SCRIPT) --save $(MODEL_FILE)

# Évaluer le modèle
evaluate:
	python3 $(TRAINING_SCRIPT) --evaluate

# Lancer le serveur FastAPI
run:
    # S'assurer que l'environnement virtuel est activé et lancer uvicorn
	. venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Pipeline complet : préparation, entraînement, sauvegarde et évaluation
pipeline: prepare train save evaluate

# Exécuter le pipeline complet et démarrer l'application
full-pipeline: pipeline run

mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
