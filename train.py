import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
# Préparer les données
def prepare_data():
    # Charger les données (ajoutez le chemin de vos données)
	data = pd.read_csv('Churn_Modelling.csv')
    
    # Appliquer les pré-traitements nécessaires (ex: nettoyage)
	data = data.dropna()  # Exemple de nettoyage des données manquantes
    
    # Séparer les caractéristiques (X) et les étiquettes (y)
	X = data.drop('Exited', axis=1)
	y = data['Exited']
    
    # Normalisation des données
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
    
    # Sauvegarder les données préparées
	joblib.dump(scaler, 'scaler.joblib')
    
    # Sauvegarder le jeu de données traité
	return X_scaled, y

# Entraînement du modèle
def train_model(X, y):
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(n_estimators=100)
	model.fit(X, y)
    
	return model

# Sauvegarder le modèle
def save_model(model, filename):
	joblib.dump(model, filename)

# Évaluation du modèle
def evaluate_model(model, X_test, y_test):
	from sklearn.metrics import classification_report
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a model for churn prediction")
	parser.add_argument('--prepare', action='store_true', help="Prepare the data")
	parser.add_argument('--train', action='store_true', help="Train the model")
	parser.add_argument('--save', action='store_true', help="Save the trained model")
	parser.add_argument('--evaluate', action='store_true', help="Evaluate the model")
	args = parser.parse_args()

	if args.prepare:
		X, y = prepare_data()

	if args.train:
		model = train_model(X, y)

	if args.save:
		save_model(model, 'classifier.joblib')

	if args.evaluate:
        # Pour évaluer, chargez un jeu de test préparé et effectuez une évaluation
		X_test, y_test = prepare_data()  # Utilisez vos données de test
		evaluate_model(model, X_test, y_test)
