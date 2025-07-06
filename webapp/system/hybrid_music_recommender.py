from implicit.als import AlternatingLeastSquares

class ALSRecommender:
    def __init__(self, interaction_matrix, track_uniques, df_music_info, als_model=None):
        self.interaction_matrix = interaction_matrix
        self.track_uniques = track_uniques
        self.df_music_info = df_music_info

        if als_model is None:
            self.als_model = AlternatingLeastSquares(factors=100, regularization=0.1, iterations=20, num_threads=1)
            self.als_model.fit(self.interaction_matrix)
        else:
            self.als_model = als_model

        self.user_index = None
        self.recommendations = None # List of tuples (track_id, energy, similarity, has been recommended)

    def make_recommendations(self, user_index, n=100):
        self.user_index = user_index

        user_items = self.interaction_matrix.tocsr()[user_index]


        top_n_recommendations_indexes, top_n_recommendations_scores = self.als_model.recommend(user_index, user_items, N=n, filter_already_liked_items=True)

        # for i in range(len(top_n_recommendations_indexes)):
        #     print(f"Track ID: {self.track_uniques[top_n_recommendations_indexes[i]]}, Similarity: {top_n_recommendations_scores[i]}")


        track_ids = self.track_uniques[top_n_recommendations_indexes].tolist()
        
        df_filtered = self.df_music_info.set_index('track_id').loc[track_ids][['energy']].reset_index()

        self.recommendations = [(track_id, energy, similarity, False) for (track_id, energy), similarity in zip(df_filtered.itertuples(index=False, name=None), top_n_recommendations_scores)]
        return self.recommendations

    
    def recommend_song(self, energy, energy_margin=0.05):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        
        closest_track_index = None
        distance_to_energy = float('inf')

        for i, (track_id, track_energy, similarity, has_been_recommended) in enumerate(self.recommendations):
            distance = abs(track_energy - energy)

            if not has_been_recommended and distance <= energy_margin:
                self.recommendations[i] = (track_id, track_energy, similarity, True)
                return (track_id, track_energy)
            
            if not has_been_recommended and distance < distance_to_energy:
                closest_track_index = i
                distance_to_energy = distance
        
        if closest_track_index is not None:
            track_id, track_energy, _, _= self.recommendations[closest_track_index]
            self.recommendations[closest_track_index] = (track_id, track_energy, similarity, True)
            return (track_id, track_energy)

        raise ValueError("All recommendations have already been recommended")


    def get_recommendations(self):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        return self.recommendations


    def get_recommendations_ids(self):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        return [track_id for track_id, _, _, _ in self.recommendations]
    
    def get_recommendations_info(self):
        track_ids = [track_id for track_id, _, _, _ in self.recommendations]
        df_ordered = self.df_music_info.set_index('track_id').loc[track_ids].reset_index()
        return df_ordered
    

class KmeansContentBasedRecommender:
    def __init__(self, id_to_cluster):
        self.id_to_cluster = id_to_cluster
        self.recommendations = None
    
    def make_cluster_recommendation(self, user_history):
        clusters = self.id_to_cluster[user_history]
        cluster_counts = clusters.value_counts()
        self.recommended_cluster = cluster_counts / len(clusters)
        return self.recommended_cluster

    def get_recommended_cluster(self):
        if self.recommended_cluster is None:
            raise ValueError("No cluster recommendation available. Please call make_cluster_recommendation first.")
        return self.recommendations
    

class HybridRecommender:
    def __init__(self, interaction_matrix, track_uniques, df_music_info, df_users, id_to_cluster, recommendations = None, als_recommender = None, content_based_recommender = None, alpha = 1):
        if als_recommender is not None:
            self.collaborative_als_recommender = als_recommender
        else:
            self.collaborative_als_recommender = ALSRecommender(interaction_matrix, track_uniques, df_music_info)
        
        if content_based_recommender is not None:
            self.content_based_recommender = content_based_recommender  
        else:
            self.content_based_recommender = KmeansContentBasedRecommender(id_to_cluster)

        self.df_music_info = df_music_info
        self.df_users = df_users
        self.id_to_cluster = id_to_cluster
        self.alpha = alpha  # Alpha is a parameter to control the influence of content-based recommendations
        self.recommendations = recommendations # List of tuples (track_id, energy, similarity, has been recommended)

    
    def make_recommendations(self, user_index, n=100):

        user_id = self.df_users['user_id'].unique()[user_index]
        user_history = self.df_users[self.df_users['user_id'] == user_id]['track_id']
        collaborative_recomendations = self.collaborative_als_recommender.make_recommendations(user_index, n)
        content_based_cluster_recommendation = self.content_based_recommender.make_cluster_recommendation(user_history)
        self.recommendations = []
        
        #We will apply a penalization to the collaborative filtering recommendation based on the user cluster preferences obtained by the content-based recommendation
        for track_id, energy, similarity, has_been_recommended in collaborative_recomendations:
            cluster_presence = 0.01 #Default multiplier. Used if the song's cluster is not in the user's cluster preferences (content-based recommendation)
            song_cluster = self.id_to_cluster[track_id]
            if song_cluster in content_based_cluster_recommendation.index:
                cluster_presence = content_based_cluster_recommendation[song_cluster]
            
            #print(track_id, song_cluster, multiplier)

            self.recommendations.append((track_id, energy, similarity * cluster_presence * self.alpha, has_been_recommended))
        self.recommendations = sorted(self.recommendations, key=lambda x: x[2], reverse=True)  # Sort new similarity


    def make_recommendations_only_collaborative(self, user_index, n=100):
        self.recommendations = self.collaborative_als_recommender.make_recommendations(user_index, n)
    
    def recommend_song(self, energy, energy_margin=0.05):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        
        closest_track_index = None
        distance_to_energy = float('inf')

        for i, (track_id, track_energy, similarity, has_been_recommended) in enumerate(self.recommendations):
            distance = abs(track_energy - energy)

            if not has_been_recommended and distance <= energy_margin:
                self.recommendations[i] = (track_id, track_energy, similarity, True)
                return (track_id, track_energy)
            
            if not has_been_recommended and distance < distance_to_energy:
                closest_track_index = i
                distance_to_energy = distance
        
        if closest_track_index is not None:
            track_id, track_energy, _, _= self.recommendations[closest_track_index]
            self.recommendations[closest_track_index] = (track_id, track_energy, similarity, True)
            return (track_id, track_energy)
    
    def get_recommendations(self):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        return self.recommendations
    
    def get_recommendations_ids(self):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        return [track_id for track_id, _, _, _ in self.recommendations]
    
    def get_recommendations_info(self):
        if self.recommendations is None:
            raise ValueError("No recommendations available. Please call make_recommendations first.")
        track_ids = [track_id for track_id, _, _, _ in self.recommendations]
        df_ordered = self.df_music_info.set_index('track_id').loc[track_ids].reset_index()
        return df_ordered