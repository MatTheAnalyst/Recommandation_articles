import streamlit as st
import requests
import asyncio
import pickle
import aiohttp

with open("liste_client.pkl", "rb") as file:
    user_ids = pickle.load(file)

url_for_recommendation = "https://ocrecommendation.azurewebsites.net/api/predict"

with open("liste_client.pkl", "rb") as file:
    user_ids = pickle.load(file)

st.title("Recommendation app")

col = st.columns([0.8, 0.2],gap="small")

with col[0]:
    user_option = st.selectbox(label="Veuillez sélectionner le numéro d'utilisateur dont vous souhaitez prédire les articles à recommander :", options=user_ids)

with col[1]:
    st.text('')
    virtual_launch = st.button(label="Faire une prédiction !")

if user_option:
    azure_function_url = url_for_recommendation + f"?user_id={user_option}"
    st.session_state['response'] = requests.get(f"{azure_function_url}", stream=True)

if virtual_launch:
    if 'response' in st.session_state:
        try:
            response = st.session_state['response'].text
            st.write(response)
        except Exception as e:
            st.write("Erreur lors de la récupération de la réponse:", str(e))
    else:
        st.write("Pas de requête en cours.")

    # Afficher un écran d'attente si la réponse n'est pas encore prête
    if 'response' not in st.session_state:
        st.write("Prédiction en cours...")

# # Requête GET à l'application mon système de recommandation hébergé avec Azure Function.
# top_5_article = f"https://ocrecommendation.azurewebsites.net/api/predict?user_id={user_option}"
# response = requests.get(top_5_article)

# launch_prediction = st.button(label="Lancer la prédiction !")
# if launch_prediction:

# # GET requests.
# response = requests.get(top_5_article)

# st.text(f"Voici le top 5 des articles recommandés pour l'utilisateur n°{user_option} :")
