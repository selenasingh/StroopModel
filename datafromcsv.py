import numpy as np
import pandas as pd

class extractRT(object):
    """
    Re-formats long-form tables to align with simulation data format. 
    """
    def __init__(self, stroopdata, emotiondata, counterbalanced):
        self.stroopdata = stroopdata
        self.emotiondata = emotiondata
        self.counterbalanced = counterbalanced
        self.final_data = pd.DataFrame([])
        self.extract_counterbalanced()

    def extract_counterbalanced(self):
        standardtrials = self.stroopdata[self.stroopdata['counterbalancing_group'] == self.counterbalanced]
        emotiontrials = self.emotiondata[self.emotiondata['counterbalancing_group'] == self.counterbalanced]

        columns_to_keep_1 = ['prolific_ID', 'congruency', 'mean_RT', 'RRS_score']
        columns_to_keep_2 = ['prolific_ID', 'valence', 'mean_RT', 'RRS_score']
        data_subset = standardtrials[columns_to_keep_1]
        data_subset_2 = emotiontrials[columns_to_keep_2]

        # Pivoting the DataFrame to reorganize mean_RT for each congruency type into separate columns
        pivoted_data = data_subset.pivot(index='prolific_ID', columns='congruency', values='mean_RT')
        pivoted_data_2 = data_subset_2.pivot(index='prolific_ID', columns='valence', values = 'mean_RT')

        # Renaming the columns for clarity
        pivoted_data.rename(columns={0: 'congruent', 1: 'incongruent'}, inplace=True)
        pivoted_data_2.rename(columns={0: 'negative', 1: 'neutral', 2: 'positive'}, inplace=True)

        # Resetting the index to make 'prolific_ID' a column again
        pivoted_data.reset_index(inplace=True)
        pivoted_data_2.reset_index(inplace=True)

        self.final_data = pivoted_data.merge(pivoted_data_2, on='prolific_ID')

        # Adding the RRS_score column by grouping it to ensure unique values for each prolific_ID
        rrs_scores = self.stroopdata[['prolific_ID', 'RRS_score', 'RRS_depression', 'RRS_brooding', 'RRS_reflection']].drop_duplicates()
        self.final_data = self.final_data.merge(rrs_scores, on='prolific_ID')
        
        self.final_data.to_csv('data/compiled_%s.csv' % self.counterbalanced)

        return self.final_data
    
stroopdata = pd.read_csv('data/stroop_standard_trials_w_subscales.csv')  
emotiondata = pd.read_csv('data/stroop_emotion_trials_w_subscales.csv')  
extractRT(stroopdata, emotiondata, 2)