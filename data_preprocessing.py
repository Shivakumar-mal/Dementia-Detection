import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Perform K-Means clustering
def perform_clustering(df, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df)
    return df

# Assign labels based on feature thresholds
def assign_labels(df):
    thresholds = {
        'Fixation_Duration': (200, 300),
        'Saccade_Velocity': (300, 700),
        'Blink_Rate': (12, 15),
        'Pursuit_Gain': (0.9, 1.1),
        'Microsaccade_Amplitude': (15, 60),
        'Microsaccade_Frequency': (1, 3),
        'Fixation_Dispersion': (1, 2),
        'Saccade_Amplitude': (5, 30),
        'Saccade_Latency': (150, 250),
        'Gaze_Path_Deviation': (1, 2)
    }

    df['Label'] = 'Normal'
    cluster_features = df.groupby('Cluster').mean()

    for feature, (low, high) in thresholds.items():
        df.loc[(df['Cluster'] == 0) & ((df[feature] < low) | (df[feature] > high)), 'Label'] = 'Dementia'

    return df

if __name__ == "__main__":
    df_combined = load_data('../data/combined_dataset.csv')
    df_combined = perform_clustering(df_combined)
    df_combined = assign_labels(df_combined)
    df_combined.to_csv('../data/labeled_dataset.csv', index=False)
