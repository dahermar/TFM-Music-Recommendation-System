import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.title("Fuzzy Logic Graphs")

bpm_antecedent = ctrl.Antecedent(np.arange(30, 201, 1), 'BPM')
bpm_variation_antecedent = ctrl.Antecedent(np.arange(-30, 31, 1), 'BPM Variation')
intensity_consequent = ctrl.Consequent(np.arange(0, 101, 1), 'Intensity')

bpm_antecedent.automf(5, names=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
bpm_variation_antecedent.automf(3, names=['Negative', 'Zero', 'Positive'])
intensity_consequent.automf(3, names=['Low', 'Medium', 'High'])

bpm_antecedent.view()
st.pyplot(plt.gcf())  # Captura la figura actual generada por view()

# Opcional: mostrar los otros también
bpm_variation_antecedent.view()
st.pyplot(plt.gcf())

intensity_consequent.view()
st.pyplot(plt.gcf())