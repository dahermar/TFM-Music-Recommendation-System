import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt


class FuzzyController:
    def __init__(self):
        self.bpm_antecedent = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Normalized BPM')
        self.bpm_variation_antecedent = ctrl.Antecedent(np.arange(-0.2, 0.21, 0.01), 'Normalized BPM Variation')
        self.energy_consequent = ctrl.Consequent(np.arange(-0.17, 1.18, 0.01), 'Energy')
        #self.energy_consequent = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Energy')

        self.bpm_antecedent['Very Light'] = fuzz.trapmf(self.bpm_antecedent.universe, [0.00, 0.00, 0.54, 0.60])
        self.bpm_antecedent['Light'] = fuzz.trapmf(self.bpm_antecedent.universe, [0.54, 0.60, 0.61, 0.67])
        self.bpm_antecedent['Moderate'] = fuzz.trapmf(self.bpm_antecedent.universe, [0.61, 0.67, 0.70, 0.84])
        self.bpm_antecedent['Vigorous'] = fuzz.trapmf(self.bpm_antecedent.universe, [0.70, 0.84, 0.93, 0.99])
        self.bpm_antecedent['Near Maximal'] = fuzz.trapmf(self.bpm_antecedent.universe, [0.93, 0.99, 1.00, 1.00])

        self.bpm_variation_antecedent['Negative'] = fuzz.trapmf(self.bpm_variation_antecedent.universe, [-0.2, -0.2, -0.15, -0.05])
        self.bpm_variation_antecedent['Zero'] = fuzz.trapmf(self.bpm_variation_antecedent.universe, [-0.15, -0.05, 0.05, 0.15])
        self.bpm_variation_antecedent['Positive'] = fuzz.trapmf(self.bpm_variation_antecedent.universe, [0.05, 0.15, 0.2, 0.21])

        
        self.energy_consequent['Low'] = fuzz.trapmf(self.energy_consequent.universe, [-0.2, -0.2, 0.0, 0.375])
        self.energy_consequent['Medium'] = fuzz.trapmf(self.energy_consequent.universe, [0.125, 0.5, 0.5, 0.875])
        self.energy_consequent['High'] = fuzz.trapmf(self.energy_consequent.universe, [0.625, 1, 1.21, 1.21])

        self.energy_consequent.defuzzify_method = 'centroid'

        # Rules:
        # Rule Intensity_zone Variation ⇒ Energy
        # R1 Very_Light – ⇒ High
        # R1 Light Negative ⇒ High
        # R1 Light Zero ⇒ High
        # R1 Moderate Negative ⇒ High

        # R2 Light Positive ⇒ Medium
        # R2 Moderate Zero ⇒ Medium
        # R2 Moderate Positive ⇒ Medium
        # R2 Vigorous Negative ⇒ Medium
        # R2 Vigorous Zero ⇒ Medium

        # R3 Near_maximal – ⇒ Low
        # R3 Vigorous Positive ⇒ Low

        rule1 = ctrl.Rule(antecedent= (self.bpm_antecedent['Very Light'] |
                        (self.bpm_antecedent['Light'] & self.bpm_variation_antecedent['Negative']) |
                        (self.bpm_antecedent['Light'] & self.bpm_variation_antecedent['Zero']) |
                        (self.bpm_antecedent['Moderate'] & self.bpm_variation_antecedent['Negative'])),
                        consequent=self.energy_consequent['High'])
        
        rule2 = ctrl.Rule(antecedent=((self.bpm_antecedent['Light'] & self.bpm_variation_antecedent['Positive']) |
                                (self.bpm_antecedent['Moderate'] & self.bpm_variation_antecedent['Zero']) |
                                (self.bpm_antecedent['Moderate'] & self.bpm_variation_antecedent['Positive']) |
                                (self.bpm_antecedent['Vigorous'] & self.bpm_variation_antecedent['Negative'])) |
                                (self.bpm_antecedent['Vigorous'] & self.bpm_variation_antecedent['Zero']),
                                consequent=self.energy_consequent['Medium'])
        
        rule3 = ctrl.Rule(antecedent=(self.bpm_antecedent['Near Maximal'] |
                        (self.bpm_antecedent['Vigorous'] & self.bpm_variation_antecedent['Positive'])),
                        consequent=self.energy_consequent['Low'])

        energy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.energy_sim = ctrl.ControlSystemSimulation(energy_ctrl)

    def calculate_energy(self, bpm, bpm_variation, age, plot_consequent=False, plot_antecedent=False):
        hr_max = 208 - 0.7 * age # Paper: Age-Predicted Maximal Heart Rate Revisited

        bpm_normalized = bpm  / hr_max
        bpm_variation_normalized = bpm_variation / hr_max

        self.energy_sim.input['Normalized BPM'] = bpm_normalized
        self.energy_sim.input['Normalized BPM Variation'] = bpm_variation_normalized
        self.energy_sim.compute()

        
        if plot_antecedent:
            st.subheader("Antecedents")
            self.bpm_antecedent.view(sim=self.energy_sim)
            st.pyplot(plt.gcf())
            plt.clf()
            self.bpm_variation_antecedent.view(sim=self.energy_sim)
            st.pyplot(plt.gcf())
            plt.clf()

        if plot_consequent:
            st.subheader("Consequent")
            self.energy_consequent.view(sim=self.energy_sim)
            st.pyplot(plt.gcf())
            plt.clf()
        
        return self.energy_sim.output['Energy']

    def view_bpm_antecedent(self):
        self.bpm_antecedent.view()
    
    def view_bpm_variation_antecedent(self):
        self.bpm_variation_antecedent.view()

    def view_energy_consequent(self):
        self.energy_consequent.view()


class EnergyCalculator:
    def __init__(self, df_gym_member, df_heart_rates, session_minute = 0, fuzzy_controller=None):
        self.user_age = df_gym_member['Age']
        self.df_heart_rates = df_heart_rates
        self.sesion_minute = session_minute
        if fuzzy_controller is None:
            self.fuzzy_controller = FuzzyController()
        else:
            self.fuzzy_controller = fuzzy_controller

    def calculate_energy(self, plot_consequent=False, plot_antecedent=False):
        if self.sesion_minute == 0:
            return 0.6, None, None # Default energy for the first song
        if self.sesion_minute >= len(self.df_heart_rates):
            return -1, None, None # Indicates that the session has ended
        bpm_current = self.df_heart_rates[self.sesion_minute]
        bpm_before = self.df_heart_rates[self.sesion_minute - 1]
        bpm_variation = bpm_current - bpm_before
        print(f"Calculating energy for session minute {self.sesion_minute}")
        print(f"Previous BPM: {self.df_heart_rates[self.sesion_minute - 1]}, Current BPM: {bpm_current}, BPM Variation: {bpm_variation}")
        return self.fuzzy_controller.calculate_energy(self.df_heart_rates[self.sesion_minute], bpm_variation, self.user_age, plot_consequent, plot_antecedent), bpm_current, bpm_before
    
    def pass_song_duration(self, song_duration=2): # Song duration in minutes
        self.sesion_minute += song_duration
        if self.sesion_minute >= len(self.df_heart_rates):
            return -1
        return self.sesion_minute
    
    def get_session_minute(self):
        return self.sesion_minute

    def view_bpm_antecedent(self):
        self.fuzzy_controller.view_bpm_antecedent()
    
    def view_bpm_variation_antecedent(self):
        self.fuzzy_controller.view_bpm_variation_antecedent()
    
    def view_energy_consequent(self):
        self.fuzzy_controller.view_energy_consequent()