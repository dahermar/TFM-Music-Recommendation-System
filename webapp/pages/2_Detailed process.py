import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import beta
import scipy.stats
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import os
import pickle

# Custom files
from system.energy_calculator import FuzzyController, EnergyCalculator
from system.hybrid_music_recommender import ALSRecommender, KmeansContentBasedRecommender, HybridRecommender
from system.two_stage_system import MusicRecommender2Stages

BASE_DIR = os.getcwd()
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
DATA_DIR = os.path.join(RESOURCES_DIR, 'data')
MODEL_DIR = os.path.join(RESOURCES_DIR, 'models')
MATRICES_DIR = os.path.join(RESOURCES_DIR, 'matrices')

# Clear cache
#st.cache_data.clear()

st.set_page_config(
    page_title="Music recommender",
    page_icon="🎵", 
    layout="centered"
)

@st.cache_data
def load_csv(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None
    
@st.cache_data
def load_cluster_mapping(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path).set_index('track_id').iloc[:, 0]
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None
    

@st.cache_data
def load_numpy_data(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None
    
@st.cache_data
def load_index_data(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return pd.Index(pd.read_csv(file_path).squeeze())
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None

@st.cache_data
def create_df_music_info(df_music_source):    
    return df_music_source[['track_id', 'name', 'artist', 'energy', 'duration_ms']]

@st.cache_data
def gym_members_count(df_gym):
    return df_gym.shape[0]


@st.cache_resource
def load_pickle(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None


st.title("Detailed recommendation process")

if 'session_started' not in st.session_state:
    st.session_state.session_started = False

if 'session_minute' not in st.session_state:
    st.session_state.session_minute = 0

if 'genarated_bpms' not in st.session_state:
    st.session_state.genarated_bpms = None


# Load data
df_gym = load_csv(DATA_DIR, 'modified_gym_members_exercise_tracking.csv')
df_heart_rates = load_csv(DATA_DIR, 'gym_members_heart_rates.csv')
df_users = load_csv(DATA_DIR, 'User Listening History_reduced.csv')
df_music = load_csv(DATA_DIR, 'Music Info.csv')

id_to_cluster = load_cluster_mapping(DATA_DIR, 'track_clusters.csv')

user_codes = load_numpy_data(DATA_DIR, 'user_codes.npy')
track_codes = load_numpy_data(DATA_DIR, 'track_codes.npy')
user_uniques = load_index_data(DATA_DIR, 'user_uniques.csv')
track_uniques = load_index_data(DATA_DIR, 'track_uniques.csv')

members_count = gym_members_count(df_gym)
df_music_info = create_df_music_info(df_music)

#Load interaction matrix
interaction_matrix_user_item = load_pickle(MATRICES_DIR, 'interaction_matrix.pkl')

#Load model
als_model = load_pickle(MODEL_DIR, 'als_model.pkl')


if df_gym is None or df_heart_rates is None or df_users is None or df_music is None or id_to_cluster is None or user_codes is None or track_codes is None or user_uniques is None or track_uniques is None:
    st.error("Error loading data files. Please check the files in the resources/data directory.")
    st.stop()

if als_model is None:
    st.error("Error loading ALS model. Please check the model in the resources/models directory.")
    st.stop()

if interaction_matrix_user_item is None:
    st.error("Error loading interaction matrix. Please check the matrix in the resources/matrices directory.")
    st.stop()

st.markdown(f"### Select your user ID")


st.write("Gym Members Exercise Dataset")
df_users_shown = df_gym[['Age', 'Gender', 'Weight (kg)','Height (m)', 'Session_Duration (hours)', 'Workout_Type']].copy()
df_users_shown.rename(columns={'Session_Duration (hours)': 'Duration (hours)'}, inplace=True) 
df_users_shown.index.name = "ID"
df_users_shown.index = range(1, len(df_users_shown) + 1)
st.dataframe(df_users_shown, height=200)

selected_user_id = st.number_input(label = ' ', min_value=1, max_value=members_count, value=1, step=1) - 1
session_button_caption = "Start session" if not st.session_state.session_started else "Restart session"

if st.session_state.session_started:
    energy_calculator = EnergyCalculator(df_gym.iloc[st.session_state.user_id], st.session_state.user_heart_rates, st.session_state.session_minute)
    als_recommender = ALSRecommender(interaction_matrix_user_item, track_uniques, df_music_info, als_model)
    hybrid_recommender = HybridRecommender(interaction_matrix_user_item, track_uniques, df_music_info, df_users, id_to_cluster, st.session_state.recommendations, als_recommender=als_recommender)
    music_recommender_2_stages = MusicRecommender2Stages(energy_calculator, hybrid_recommender, st.session_state.user_id, df_music_info)
    st.markdown(f"### Welcome user {st.session_state.user_id + 1}")
    st.write("User listened songs")
    st.dataframe(st.session_state.listened_songs)
    st.write("Recommended Songs")
    st.dataframe(music_recommender_2_stages.get_recommendations_info())

if st.button(session_button_caption):
    st.session_state.user_id = selected_user_id
    st.session_state.session_minute = 0
    st.session_state.user_heart_rates = df_heart_rates[df_heart_rates['User_ID'] == st.session_state.user_id]['Heart_Rate'].tolist()
    st.session_state.session_started = True

    user_listened_songs = df_users[df_users['user_id'] == user_uniques[st.session_state.user_id]].track_id
    st.session_state.listened_songs = df_music_info[df_music_info['track_id'].isin(user_listened_songs)]

    als_recommender = ALSRecommender(interaction_matrix_user_item, track_uniques, df_music_info, als_model)
    hybrid_recommender = HybridRecommender(interaction_matrix_user_item, track_uniques, df_music_info, df_users, id_to_cluster, als_recommender=als_recommender)
    music_recommender_2_stages = MusicRecommender2Stages(None, hybrid_recommender, st.session_state.user_id, df_music_info)
    music_recommender_2_stages.make_recommendations(n=100)
    st.session_state.recommendations = music_recommender_2_stages.get_recommendations()
    st.rerun()

if st.session_state.session_started:
    if st.button('Pass time'):
        minute, df_recommended_song, energy, bpm_current, bpm_before = music_recommender_2_stages.recommend_song(plot_antecedent=True, plot_consequent=True)
        st.session_state.session_minute = music_recommender_2_stages.get_session_minute()
        st.markdown(f"### Session minute: {minute}")
        if df_recommended_song is None:
            st.markdown("##### Session ended")
        else:
            if minute == 0:
                st.markdown("##### Warm-up song")
            else:
                st.write(f"Current bpm: {int(bpm_current)}, Last bpm: {int(bpm_before)}")
            st.markdown(f"##### Energy level for next song: {energy}")
            
            if energy < 0.2:
                st.markdown("##### Decrease training intensity significantly")
            elif energy < 0.4:
                st.markdown("##### Decrease training intensity slightly")
            elif energy < 0.6:
                st.markdown("##### Maintain training intensity")
            elif energy < 0.8:
                st.markdown("##### Increase training intensity slightly")
            else:
                st.markdown("##### Increase training intensity significantly")
            st.markdown("##### Recommended song")
            st.dataframe(df_recommended_song.assign(duration_ms=lambda df: df['duration_ms'].apply(lambda x: f"{int(x // 60000)}:{int((x % 60000) // 1000):02}")))



    if st.button('End session'):
        st.session_state.session_started = False
        st.session_state.session_minute = 0
        st.session_state.genarated_bpms = None
        st.rerun()
