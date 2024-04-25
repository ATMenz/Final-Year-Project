import os
import pandas as pd

# Configuration
output_file = r"output.xlsx"
column = "Sentiment_Score"
input_folder = r"worldnews\worldnews_processed_economics"

def process_file(filepath):
    df = pd.read_excel(filepath)
    df['Created_UTC'] = pd.to_datetime(df['Created_UTC'])
    df['Year'] = df['Created_UTC'].dt.year
    df['Month'] = df['Created_UTC'].dt.month
    monthly_yearly_avg = df.groupby(['Year', 'Month'])[column].mean().reset_index()
    return monthly_yearly_avg

def process_subfolder(subfolder_path):
    all_sentiment_scores = []
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(subfolder_path, filename)
            monthly_yearly_avg = process_file(filepath)
            all_sentiment_scores.append(monthly_yearly_avg)
    total_avg = pd.concat(all_sentiment_scores).groupby(['Year', 'Month']).mean().reset_index()
    return total_avg

def save_data():
    processed_data = process_subfolder(input_folder)
    processed_data.to_excel(output_file, index=False)

if __name__ == "__main__":
    save_data()