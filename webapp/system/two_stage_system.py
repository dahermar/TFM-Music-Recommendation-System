class MusicRecommender2Stages:
    def __init__(self, energy_calculator, hybrid_recommender, user_index, df_music_info):
        self.energy_calculator = energy_calculator
        self.hybrid_recommender = hybrid_recommender
        self.user_index = user_index
        self.df_music_info = df_music_info


    def make_recommendations(self, n=100):
        self.hybrid_recommender.make_recommendations(self.user_index, n)
        
    
    def recommend_song(self, plot_consequent=False, plot_antecedent=False):
        current_minute = self.energy_calculator.get_session_minute()
        energy, bpm_current, bpm_before = self.energy_calculator.calculate_energy(plot_consequent, plot_antecedent)
        if energy == -1:
            return current_minute, None, None, None, None # Session has ended
        print(f"Energy level needed for recommendation: {energy}")
        song_id, _ = self.hybrid_recommender.recommend_song(energy)
        song_duration_minutes = self.df_music_info[self.df_music_info['track_id'] == song_id]['duration_ms'].values[0] // 60000
        self.energy_calculator.pass_song_duration(song_duration_minutes)
        return current_minute, self.df_music_info[self.df_music_info['track_id'] == song_id], energy, bpm_current, bpm_before
    
    def pass_song_duration(self, song_duration=2):
        return self.energy_calculator.pass_song_duration(song_duration)
    
    def get_session_minute(self):
        return self.energy_calculator.get_session_minute()
    
    def get_recommendations(self):
        return self.hybrid_recommender.get_recommendations()
    
    def get_recommendations_ids(self):
        return self.hybrid_recommender.get_recommendations_ids()
    
    def get_recommendations_info(self):
        return self.hybrid_recommender.get_recommendations_info()
    

    