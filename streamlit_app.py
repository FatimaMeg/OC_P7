import joblib
import re
#from sklearn.neural_network import MLPClassifier
#from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import pickle
import matplotlib.pyplot as plt

st.write("# Dashboard Credit scoring")

PAYMENT_RATE = st.number_input("var1", value = 6.08882521e-02)
EXT_SOURCE_2 = st.number_input("var2", value = 4.31669488e-01)
EXT_SOURCE_3 = st.number_input("var3", value = 3.72333666e-01)
DAYS_BIRTH = st.number_input("var4", value = -1.93110000e+04)
AMT_ANNUITY = st.number_input("var5", value = 1.72125000e+04)
BURO_DAYS_CREDIT_MAX = st.number_input("var6", value = -5.00000000e+02)
BURO_DAYS_CREDIT_ENDDATE_MAX = st.number_input("var7", value = 1.04270000e+04)
DAYS_EMPLOYED = st.number_input("var8", value = -5.70000000e+02)
AMT_GOODS_PRICE = st.number_input("var9", value = 2.02500000e+05)
DAYS_REGISTRATION = st.number_input("var10", value = -8.77100000e+03)
ANNUITY_INCOME_PERC = st.number_input("var11", value = 1.53000000e-01)
PREV_CNT_PAYMENT_MEAN = st.number_input("var12", value = 1.25714286e+01)
INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.number_input("var13", value = -2.40000000e+01)
DAYS_ID_PUBLISH = st.number_input("var14", value = -2.84100000e+03)
APPROVED_CNT_PAYMENT_MEAN = st.number_input("var15", value = 1.13333333e+01)
BURO_AMT_CREDIT_SUM_SUM = st.number_input("var16", value = 1.20145500e+06)
INSTAL_DPD_MEAN = st.number_input("var17", value = 1.42105263e+00)
INSTAL_AMT_PAYMENT_MIN = st.number_input("var18", value = 1.29465000e+02)
REGION_POPULATION_RELATIVE = st.number_input("var19", value = 4.62200000e-02)
BURO_AMT_CREDIT_SUM_DEBT_MEAN = st.number_input("var20", value = 6.13023750e+04)


model = joblib.load('pipeline_bank_lgbm.joblib')

def client_scoring(model, data):
#[var1,var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20]
	faillite_resultat = model.predict([data])[0]
	faillite_proba = model.predict_proba([data])

	return {'Le client risque t il l d etre en faillite' : faillite_resultat,\
			'Sa probabilite de faillite est de ': faillite_proba[0][1]}

if PAYMENT_RATE != '':
	my_data = [PAYMENT_RATE, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, AMT_ANNUITY,\
									BURO_DAYS_CREDIT_MAX, BURO_DAYS_CREDIT_ENDDATE_MAX,\
									DAYS_EMPLOYED, AMT_GOODS_PRICE,\
									DAYS_REGISTRATION, ANNUITY_INCOME_PERC, \
									PREV_CNT_PAYMENT_MEAN, INSTAL_DAYS_ENTRY_PAYMENT_MAX,\
									DAYS_ID_PUBLISH, APPROVED_CNT_PAYMENT_MEAN,\
									BURO_AMT_CREDIT_SUM_SUM,INSTAL_DPD_MEAN,\
									INSTAL_AMT_PAYMENT_MIN, REGION_POPULATION_RELATIVE, \
									BURO_AMT_CREDIT_SUM_DEBT_MEAN]
	result = client_scoring(model, my_data)

	st.write(result)

	
	explain_pred = st.button('Explain Predictions')

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

	pickle_in = open("X_train_Nono2.pkl", "rb")
	donnees_train = pickle.load(pickle_in)
	pickle_in.close()

	#st.write(donnees_train[0])


	if explain_pred:
		with st.spinner('Generating explanations'):
			#pickle_in = open("X_train_Nono.pkl", "rb")
			#donnees_train = pickle.load(pickle_in)
			explainer = lime_tabular.LimeTabularExplainer(donnees_train,mode="classification",class_names=features)
			#explainer = LimeTextExplainer(class_names=class_names)
			exp = explainer.explain_instance(donnees_train[50],
				model.predict_proba, num_features=20)
			#components.html(exp.as_html())
			st.pyplot(exp.as_pyplot_figure())
		#st.write("en cours de reflexion...")
