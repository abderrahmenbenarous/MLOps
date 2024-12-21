import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
# Préparer les données
def prepare_data():
	encoder = LabelEncoder()
    # Charger les données (ajoutez le chemin de vos données)
	data = pd.read_csv('Churn_Modelling.csv')
    # Appliquer les pré-traitements nécessaires (ex: nettoyage)
	data = data.drop(['Surname','Geography'],axis=1)
	data = data.dropna()  # Exemple de nettoyage des données manquantes
    # Séparer les caractéristiques (X) et les étiquettes (y)
	data['Gender'] = encoder.fit_transform(data['Gender'])
	X = data.drop('Exited', axis=1)
	X.drop(columns=['RowNumber','CustomerId',], inplace=True)
	y = data['Exited']
	x_train , x_test , y_train , y_test = train_test_split(X,y,test_size =0.2,random_state=1)
    # Normalisation des données
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.fit_transform(x_test)
    # Sauvegarder les données préparées
	joblib.dump(scaler, 'scaler.joblib')
	joblib.dump({"x_train_scaled": x_train_scaled,"y_train": y_train,"x_test_scaled": x_test_scaled,"y_test": y_test}, "data_prepared.joblib")
    # Sauvegarder le jeu de données traité
	return x_train, y_train , x_test , y_test
# Entraînement du modèle
def train_model(X, y):
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(n_estimators=100,random_state=42)
	model.fit(X, y)
	return model

# Sauvegarder le modèle
def save_model(model, filename):
	joblib.dump(model, filename)
# Évaluation du modèle
def evaluate_model(model, X_test, y_test):
	from sklearn.metrics import classification_report
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(classification_report(y_test, y_pred))
	return accuracy
def train_and_log():
	mlflow.set_experiment("Customer Churn Prediction")
	with mlflow.start_run() as run:
	# Préparer les données
		x_train, y_train, x_test, y_test = prepare_data()
        # Entraîner le modèle
		model = train_model(x_train, y_train)
        # Sauvegarder le modèle
		save_model(model, 'classifier.joblib')

        # Évaluer le modèle
		accuracy = evaluate_model(model, x_test, y_test)
		   # Définir le schéma pour le modèle
		input_schema = Schema([
		ColSpec("double", "CreditScore"),
		ColSpec("string", "Gender"),
		ColSpec("integer", "Age"),
		ColSpec("integer", "Tenure"),
		ColSpec("double", "Balance"),
		ColSpec("integer", "NumOfProducts"),
		ColSpec("boolean", "HasCrCard"),
		ColSpec("boolean", "IsActiveMember"),
		ColSpec("double", "EstimatedSalary")])

		output_schema = Schema([ColSpec("integer", "Exited")])
		# Créer la signature du modèle
		signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        # Suivi avec MLflow
		mlflow.log_param("n_estimators", 100)
		mlflow.log_metric("accuracy", accuracy)
		mlflow.sklearn.log_model(model, "model",signature=signature)
		# Enregistrer des tags et une description
		mlflow.set_tags({
		"model_name": "CustomerChurnModel",
		"version": "1.0",
		"description": "This model predicts customer churn based on historical data.",
		"author": "Abderrahmen Benarous"
		})
		print("Modèle enregistré avec MLflow.")
		run_id = run.info.run_id
		if  accuracy > 0.8:
			model_name = "Churn_Prediction_Model"
			mlflow.register_model(f"runs:/{run_id}/model",model_name)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a model for churn prediction")
	parser.add_argument('--prepare', action='store_true', help="Prepare the data")
	parser.add_argument('--train', action='store_true', help="Train the model")
	parser.add_argument('--save', action='store_true', help="Save the trained model")
	parser.add_argument('--evaluate', action='store_true', help="Evaluate the model")
	args = parser.parse_args()

	if args.prepare:
		x_train_scaled,y_train,x_test_scaled,y_test = prepare_data()

	if args.train:
		train_and_log()
	if args.save:
		save_model(model, 'classifier.joblib')

	if args.evaluate:
        # Pour évaluer, chargez un jeu de test préparé et effectuez une évaluation
		x_train_scaled, y_train, X_test, y_test = prepare_data()  # Utilisez vos données de test
		model=joblib.load("classifier.joblib")
		evaluate_model(model, X_test, y_test)
