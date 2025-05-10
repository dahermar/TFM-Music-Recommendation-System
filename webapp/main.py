import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import beta
import scipy.stats
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Music recommender",
    page_icon="🎵", 
    layout="centered"
)

st.title("Music recommendation system")



# TODO Limpiar el datast en otro archivo
df_gym = pd.read_csv('../data/Gym Members Exercise Dataset/gym_members_exercise_tracking.csv')
raw_df = pd.read_csv('../data/Spotify-30ksongs/spotify_songs.csv')
df_spotify = raw_df.dropna()

#Generate synthetic data
#TODO Tenerla generada ya en otro archivo


def my_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

def genarate_heart_rates_beta(min_val, max_val, avg, std, N):
    my_dist = my_distribution(min_val, max_val, avg, std)
    
    generated_numbers = my_dist.rvs(size=N)
    if True:  
        x = np.linspace(min_val, max_val, 100)
        y = my_dist.pdf(x)
        df_distr = pd.DataFrame({'x': x, 'pdf': y})
        df_distr.set_index('x', inplace=True)
        
    return generated_numbers, df_distr

mean_std_ratio_avg_max = 0.36578887474025085 #Obtained from doing the mean

min_val, max_val, avg, N = df_gym.iloc[0]['Resting_BPM'], df_gym.iloc[0]['Max_BPM'], df_gym.iloc[0]['Avg_BPM'], math.trunc(df_gym.iloc[0]['Session_Duration (hours)'] * 60)
std = mean_std_ratio_avg_max * (max_val - avg)

genarated_heart_rates_beta, df_distr = genarate_heart_rates_beta(min_val, max_val, avg, std, N)

# st.write('Real data')
# st.write(f"Min: {min_val}, Max: {max_val}, Avg: {avg}, Std: {std}, N: {N}")

# st.write("Generated data using beta distribution")
# st.write(f"Generated average: {np.mean(genarated_heart_rates_beta):.2f}")
# st.write(f"Generated std: {np.std(genarated_heart_rates_beta, ddof=1):.2f}")
# st.write(f"Generated min: {np.min(genarated_heart_rates_beta):.2f}")
# st.write(f"Generated max: {np.max(genarated_heart_rates_beta):.2f}")

# st.line_chart(df_distr)

#First model - Fuzzy logic
bpm_antecedent = ctrl.Antecedent(np.arange(30, 201, 1), 'BPM')
bpm_variation_antecedent = ctrl.Antecedent(np.arange(-30, 31, 1), 'BPM Variation')
intensity_consequent = ctrl.Consequent(np.arange(0, 101, 1), 'Intensity')

bpm_antecedent.automf(5, names=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
bpm_variation_antecedent.automf(3, names=['Negative', 'Zero', 'Positive'])
intensity_consequent.automf(3, names=['Low', 'Medium', 'High'])

#Rules
rule1 = ctrl.Rule(antecedent= (bpm_antecedent['Very Low'] |
                        (bpm_antecedent['Low'] & bpm_variation_antecedent['Negative']) |
                        (bpm_antecedent['Low'] & bpm_variation_antecedent['Zero']) |
                        (bpm_antecedent['Medium'] & bpm_variation_antecedent['Negative'])),
                        consequent=intensity_consequent['High'])
rule2 = ctrl.Rule(antecedent=((bpm_antecedent['Low'] & bpm_variation_antecedent['Positive']) |
                        (bpm_antecedent['Medium'] & bpm_variation_antecedent['Zero']) |
                        (bpm_antecedent['High'] & bpm_variation_antecedent['Negative'])),
                        consequent=intensity_consequent['Medium'])
rule3 = ctrl.Rule(antecedent=((bpm_antecedent['Medium'] & bpm_variation_antecedent['Positive']) |
                        (bpm_antecedent['High'] & bpm_variation_antecedent['Zero']) |
                        (bpm_antecedent['High'] & bpm_variation_antecedent['Positive'])),
                        consequent=intensity_consequent['Low'])

#Controller
intensity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
intensity_sim = ctrl.ControlSystemSimulation(intensity_ctrl)

def calculate_intensity_fuzzy(bpm, bpm_variation, plot_consequent=False, plot_antecedent=False):
    intensity_sim.input['BPM'] = bpm
    intensity_sim.input['BPM Variation'] = bpm_variation
    intensity_sim.compute()
    if plot_consequent:
        intensity_consequent.view(sim=intensity_sim)
        st.pyplot(plt.gcf())
    if plot_antecedent:
        bpm_antecedent.view(sim=intensity_sim)
        st.pyplot(plt.gcf())
        bpm_variation_antecedent.view(sim=intensity_sim)
        st.pyplot(plt.gcf())
    return intensity_sim.output['Intensity']


st.write('### Example')
st.markdown("""
- **BPM**: 120
- **BPM Variation**: 10
""")

calculate_intensity_fuzzy(120, 10, plot_consequent=False, plot_antecedent=False)