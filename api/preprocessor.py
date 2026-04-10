import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class StormDamagePreprocessor:
    def __init__(self, models_dir='../models'):
        # Load all label encoders from a single pickle
        with open(f'{models_dir}/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            self.event_type_le = encoders['event_type_le']
            self.state_encoder = encoders['state_encoder']
            self.cz_name_le = encoders['cz_name_le']
            self.flood_cause_le = encoders['flood_cause_le']
            self.cz_type_encoder = encoders['cz_type_encoder']
            self.source_le = encoders['source_le']
        
        with open(f'{models_dir}/magnitude_median.pkl', 'rb') as f:
            self.magnitude_median = pickle.load(f)
        
        with open(f'{models_dir}/state_avg_damage.pkl', 'rb') as f:
            self.state_avg_damage = pickle.load(f)
        
        with open(f'{models_dir}/global_avg_damage.pkl', 'rb') as f:
            self.global_avg = pickle.load(f)
        
        with open(f'{models_dir}/keep_embeds.pkl', 'rb') as f:
            self.keep_embeds = pickle.load(f)
        
        # feature_config stores column order and sentence transformer model name
        with open(f'{models_dir}/feature_config.pkl', 'rb') as f:
            self.feature_config = pickle.load(f)
        
        self.sentence_model = SentenceTransformer(
            self.feature_config['sentence_transformer_model']
        )
    
    def preprocess(self, input_data):
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
            'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
            'November': 11, 'December': 12
        }
        month_name = input_data.get('MONTH_NAME', 'January')
        month_num = month_mapping.get(month_name, 1)
        
        # Map EF scale strings to integers; default to 0 for missing or non-tornado events
        tor_f_scale = input_data.get('TOR_F_SCALE', 0)
        if tor_f_scale == 'EF0': tor_f_scale = 0
        elif tor_f_scale == 'EF1': tor_f_scale = 1
        elif tor_f_scale == 'EF2': tor_f_scale = 2
        elif tor_f_scale == 'EF3': tor_f_scale = 3
        elif tor_f_scale == 'EF4': tor_f_scale = 4
        elif tor_f_scale == 'EF5': tor_f_scale = 5
        else: tor_f_scale = 0
        
        tor_length = input_data.get('TOR_LENGTH', 0) or 0
        tor_width = input_data.get('TOR_WIDTH', 0) or 0
        
        # Fall back to 0 for unseen categories rather than raising an error
        event_type = input_data.get('EVENT_TYPE', 'Unknown')
        try:
            event_type_label = self.event_type_le.transform([event_type])[0]
        except:
            event_type_label = 0
        
        state = input_data.get('STATE', 'TEXAS')
        try:
            state_label = self.state_encoder.transform([state])[0]
        except:
            state_label = 0
        
        # Fill missing magnitude with the training median
        magnitude = input_data.get('MAGNITUDE')
        if magnitude is None or pd.isna(magnitude):
            magnitude = self.magnitude_median
        
        cz_name = input_data.get('CZ_NAME', 'Unknown')
        try:
            cz_name_label = self.cz_name_le.transform([cz_name])[0]
        except:
            cz_name_label = 0
        
        flood_cause = input_data.get('FLOOD_CAUSE', 'Unknown') or 'Unknown'
        try:
            flood_cause_encoded = self.flood_cause_le.transform([flood_cause])[0]
        except:
            flood_cause_encoded = 0
        
        cz_type = input_data.get('CZ_TYPE', 'C')
        try:
            cz_type_encoded = self.cz_type_encoder.transform([cz_type])[0]
        except:
            cz_type_encoded = 0
        
        source = input_data.get('SOURCE', 'Public')
        try:
            source_encoded = self.source_le.transform([source])[0]
        except:
            source_encoded = 0
        
        # Combine both narratives and embed; then select only the dimensions kept during training
        episode_narrative = input_data.get('EPISODE_NARRATIVE', '')
        event_narrative = input_data.get('EVENT_NARRATIVE', '')
        combined_narrative = f"{episode_narrative} {event_narrative}"
        
        full_embeddings = self.sentence_model.encode([combined_narrative])[0]
        
        selected_embeddings = {}
        for embed_col in self.keep_embeds:
            embed_idx = int(embed_col.replace('embed_', ''))
            selected_embeddings[embed_col] = full_embeddings[embed_idx]
        
        # Use state average damage as a feature; fall back to global average for unseen states
        if state_label in self.state_avg_damage.index:
            state_avg_dp = self.state_avg_damage[state_label]
        else:
            state_avg_dp = self.global_avg
        
        # state_avg_dp is appended last to match the column order used during training
        features = {
            'MONTH_NAME': month_num,
            'INJURIES_DIRECT': input_data.get('INJURIES_DIRECT', 0),
            'INJURIES_INDIRECT': input_data.get('INJURIES_INDIRECT', 0),
            'DEATHS_DIRECT': input_data.get('DEATHS_DIRECT', 0),
            'DEATHS_INDIRECT': input_data.get('DEATHS_INDIRECT', 0),
            'SOURCE': source_encoded,
            'MAGNITUDE': magnitude,
            'TOR_F_SCALE': tor_f_scale,
            'TOR_LENGTH': tor_length,
            'TOR_WIDTH': tor_width,
            'DURATION_MINUTES': input_data.get('DURATION_MINUTES', 0),
            'event_type_label': event_type_label,
            'state_label': state_label,
            'cz_name_label': cz_name_label,
            'FLOOD_CAUSE_ENCODED': flood_cause_encoded,
            'CZ_TYPE_ENCODED': cz_type_encoded,
            **selected_embeddings,
            'state_avg_dp': state_avg_dp
        }
        
        df = pd.DataFrame([features])
        column_order = self.feature_config['feature_columns'] + ['state_avg_dp']
        df = df[column_order]
        
        return df.to_numpy()
