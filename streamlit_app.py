import joblib
#from sklearn.neural_network import MLPClassifier
#from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import pickle
import matplotlib.pyplot as plt

st.write("# Dashboard Credit scoring")

NUM_CLIENT = st.number_input("Renseigner ci-dessous le numéro de client pour obtenir"
							 " le dashboard associé à ce client", value= 456202)

model = joblib.load('pipeline_bank_lgbm.joblib')

#On récupère les deux fichiers pickle utiles : le fichier des données clients et celui des données train
file_X_train = open("X_train_Nono2.pkl", "rb")
donnees_train = pickle.load(file_X_train)
file_X_train.close()

file_clients = open("fichierClient.pkl", "rb")
donnees_clients = pickle.load(file_clients)
file_clients.close()

#On crée ici notre variable features, qu'on pourrait mettre aussi dans un pkl
features = ['PAYMENT_RATE',
 				'EXT_SOURCE_2',
 				'EXT_SOURCE_3',
 				'DAYS_BIRTH',
 				'AMT_ANNUITY',
 				'BURO_DAYS_CREDIT_MAX',
 				'BURO_DAYS_CREDIT_ENDDATE_MAX',
 				'DAYS_EMPLOYED',
 				'AMT_GOODS_PRICE',
 				'DAYS_REGISTRATION',
 				'ANNUITY_INCOME_PERC',
 				'PREV_CNT_PAYMENT_MEAN',
 				'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
 				'DAYS_ID_PUBLISH',
 				'APPROVED_CNT_PAYMENT_MEAN',
 				'BURO_AMT_CREDIT_SUM_SUM',
 				'INSTAL_DPD_MEAN',
 				'INSTAL_AMT_PAYMENT_MIN',
 				'REGION_POPULATION_RELATIVE',
 				'BURO_AMT_CREDIT_SUM_DEBT_MEAN']

#On définit notre fonction prediction qui prend en entrée toutes les features
def num_client_scoring(model, num_client):
	#On récupère d'abord toutes les features associées à ce numéro client pour les entrer dans le model
	data2 = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == num_client, features]
	faillite_resultat2 = model.predict(data2)[0]
	faillite_proba2 = model.predict_proba(data2)

	return {'Le client risque-t-il d\'être en faillite' : faillite_resultat2,\
			'Sa probabilite de faillite est de ': round(faillite_proba2[0][1]*100,4) \'%'}



#On récupère les features de notre client
data3 = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == NUM_CLIENT, features]

if NUM_CLIENT != '':

	#ligne test qui permet d'afficher le dataframe en cas de tests unitaires
	# st.dataframe(data3)

	result2 = num_client_scoring(model, NUM_CLIENT)

	st.write(result2)


	explain_pred = st.button('Explain Predictions')

	#st.write(donnees_train[0])


	if explain_pred:
		with st.spinner('Generating explanations'):
			explainer = lime_tabular.LimeTabularExplainer(donnees_train,mode="classification",class_names=features)
			#explainer = LimeTextExplainer(class_names=class_names)
			exp = explainer.explain_instance(data3.values[0],
				model.predict_proba, num_features=20)
			#components.html(exp.as_html())
			st.pyplot(exp.as_pyplot_figure())
		#st.write("en cours de reflexion...")
