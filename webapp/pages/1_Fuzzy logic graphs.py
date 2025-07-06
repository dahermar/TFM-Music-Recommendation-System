import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skfuzzy")

st.set_page_config(
    page_title="Music recommender",
    page_icon="🎵", 
    layout="centered",
)

st.title("Stage 1 - Fuzzy Logic System")



bpm_antecedent = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Normalized BPM')
bpm_variation_antecedent = ctrl.Antecedent(np.arange(-0.2, 0.21, 0.01), 'Normalized BPM Variation')
energy_consequent = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Energy')

bpm_antecedent['Very Light'] = fuzz.trapmf(bpm_antecedent.universe, [0.00, 0.00, 0.54, 0.60])
bpm_antecedent['Light'] = fuzz.trapmf(bpm_antecedent.universe, [0.54, 0.60, 0.61, 0.67])
bpm_antecedent['Moderate'] = fuzz.trapmf(bpm_antecedent.universe, [0.61, 0.67, 0.74, 0.80])
bpm_antecedent['Vigorous'] = fuzz.trapmf(bpm_antecedent.universe, [0.74, 0.80, 0.93, 0.99])
bpm_antecedent['Near Maximal'] = fuzz.trapmf(bpm_antecedent.universe, [0.93, 0.99, 1.00, 1.00])

bpm_variation_antecedent['Negative'] = fuzz.trapmf(bpm_variation_antecedent.universe, [-0.2, -0.2, -0.15, -0.05])
bpm_variation_antecedent['Zero'] = fuzz.trapmf(bpm_variation_antecedent.universe, [-0.15, -0.05, 0.05, 0.15])
bpm_variation_antecedent['Positive'] = fuzz.trapmf(bpm_variation_antecedent.universe, [0.05, 0.15, 0.2, 0.21])

energy_consequent['Low'] = fuzz.trapmf(energy_consequent.universe, [0.0, 0.0, 0.0, 0.5])
energy_consequent['Medium'] = fuzz.trapmf(energy_consequent.universe, [0.25, 0.5, 0.5, 0.75])
energy_consequent['High'] = fuzz.trapmf(energy_consequent.universe, [0.5, 1, 1.0, 1.0])

st.subheader("Antecedents")

bpm_antecedent.view()
plt.gcf().set_size_inches(3.1, 2.4) #Default size is 6.4, 4.8
plt.legend(fontsize='x-small') # Default fontsize is medium
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlabel("Normalized BPM", fontsize='x-small')
plt.ylabel("Membership", fontsize='x-small')
st.pyplot(plt.gcf())


bpm_variation_antecedent.view()
plt.gcf().set_size_inches(3.1, 2.4) #Default size is 6.4, 4.8
plt.legend(fontsize='x-small') # Default fontsize is medium
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlabel("Normalized BPM Variation", fontsize='x-small')
plt.ylabel("Membership", fontsize='x-small')
st.pyplot(plt.gcf())

st.subheader("Consequent")

energy_consequent.view()
plt.gcf().set_size_inches(3.1, 2.4) #Default size is 6.4, 4.8
plt.legend(fontsize='x-small') # Default fontsize is medium
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlabel("Energy", fontsize='x-small')
plt.ylabel("Membership", fontsize='x-small')
st.pyplot(plt.gcf())

st.subheader("Rule Base")

rules_data = [
    ["R1", "Very Light", "–", "⇒", "High"],
    ["R1", "Light", "Negative", "⇒", "High"],
    ["R1", "Light", "Zero", "⇒", "High"],
    ["R1", "Moderate", "Negative", "⇒", "High"],
    ["R2", "Light", "Positive", "⇒", "Medium"],
    ["R2", "Moderate", "Zero", "⇒", "Medium"],
    ["R2", "Moderate", "Positive", "⇒", "Medium"],
    ["R2", "Vigorous", "Negative", "⇒", "Medium"],
    ["R2", "Vigorous", "Zero", "⇒", "Medium"],
    ["R3", "Near Maximal", "–", "⇒", "Low"],
    ["R3", "Vigorous", "Positive", "⇒", "Low"]
]

rules_df = pd.DataFrame(
    rules_data, 
    columns=["Rule", "Intensity Zone (BPM)", "Variation", "⇒", "Energy"]
)

for rule in ["R1", "R2", "R3"]:
    st.text(f"Rule {rule}")
    st.dataframe(
        rules_df[rules_df["Rule"] == rule].drop(columns=["Rule"]).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )