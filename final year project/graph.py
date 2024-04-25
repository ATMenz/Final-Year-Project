import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from pandas import to_datetime
from matplotlib.dates import MonthLocator, DateFormatter, date2num
import matplotlib.dates as mdates
matplotlib.use('Agg')


# Configuration
input_folder = r"worldnews\worldnews_processed_economics"
output_folder = os.path.join("graphs", f"{input_folder}_graphs")

Path(output_folder).mkdir(parents=True, exist_ok=True)
combined_output_folder = os.path.join(output_folder, "combined")
emotion_pie_folder = os.path.join(output_folder, "emotion_pie")
sentiment_output_folder = os.path.join(output_folder, "sentiment")
Path(sentiment_output_folder).mkdir(parents=True, exist_ok=True)
Path(emotion_pie_folder).mkdir(parents=True, exist_ok=True)
Path(combined_output_folder).mkdir(parents=True, exist_ok=True)
print(f"Directory '{combined_output_folder}' exists after creation: {Path(combined_output_folder).exists()}")

emotion_colors = {
    'trust': '#1f77b4',
    'fear': '#ff7f0e',
    'anticipation': '#2ca02c',
    'anger': '#d62728',
    'sadness': '#9467bd',
    'joy': '#8c564b',
    'surprise': '#e377c2',
    'disgust': '#7f7f7f',
    'other': '#bcbd22',  
    'No emotion': '#c7c7c7'    
}

all_years_data = pd.DataFrame()
all_years_weighted_data = pd.DataFrame()
all_years_non_weighted_data = pd.DataFrame()


def plot_standard_deviation_over_time(input_folder, output_folder, weighted=True):
    global all_years_weighted_data
    global all_years_non_weighted_data
    
    previous_year_avg_std = None
    
    print("Starting plot generation...")
    for file in sorted(os.listdir(input_folder)):
        if file.endswith(".xlsx"):
            print(f"Processing file: {file}")
            try:
                df = pd.read_excel(os.path.join(input_folder, file))
            except Exception as e:
                print(f"Error reading file {file}: {e}. Skipping...")
                continue
            year = file.split(".")[0]
            df.columns = [x.lower() for x in df.columns]
            std_col = 'weighted_monthly_std' if weighted else 'unweighted_monthly_std'
            if 'created_utc' in df.columns and std_col in df.columns:
                df['created_utc'] = pd.to_datetime(df['created_utc'])
                df['month'] = df['created_utc'].dt.to_period('M').dt.strftime('%Y-%m')
                monthly_std = df.groupby('month')[std_col].mean().reset_index()
                monthly_std['year'] = year
                if weighted:
                    all_years_weighted_data = pd.concat([all_years_weighted_data, monthly_std], ignore_index=True)
                else:
                    all_years_non_weighted_data = pd.concat([all_years_non_weighted_data, monthly_std], ignore_index=True)

                plt.figure(figsize=(10, 6))
                plt.plot(monthly_std['month'], monthly_std[std_col], marker='o', linestyle='-')
                
                plt.ylim(0, 1)
                title = f'Weighted Standard Deviation Over Time in {year}' if weighted else f'Non-Weighted Standard Deviation Over Time in {year}'
                plt.title(title)
                plt.xlabel('Month')
                plt.xticks(rotation=90)
                plt.ylabel(std_col.replace('_', ' ').title())
                plt.tight_layout()
                plot_filename = os.path.join(output_folder, f"{year}_{std_col}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Graph saved: {plot_filename}")
            else:
                print(f"Required columns missing in {file}")
    
    combined_data = all_years_weighted_data if weighted else all_years_non_weighted_data
    if not combined_data.empty:
        yearly_averages = combined_data.groupby('year')[std_col].mean()
        yearly_changes = yearly_averages.pct_change() * 100

        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        legend_labels_with_colors = {}

        for year in combined_data['year'].unique():
            yearly_data = combined_data[combined_data['year'] == year]
            plt.plot(yearly_data['month'],
                    yearly_data[std_col],
                    marker='o', linestyle='-', label=f"{year}")

            avg_std = yearly_averages[year]
            if year != combined_data['year'].min():
                change = yearly_changes[year]
                text_color = 'green' if change > 0 else 'red'
                legend_labels_with_colors[year] = (f"{year} - {avg_std:.3f} ({change:+.2f}%)", text_color)
            else:
                legend_labels_with_colors[year] = (f"{year} - {avg_std:.3f}", 'black')

        handles, labels = ax.get_legend_handles_labels()
        new_labels = [legend_labels_with_colors[label][0] for label in labels]
        new_label_colors = [legend_labels_with_colors[label][1] for label in labels]
        new_legend = plt.legend(handles, new_labels, title='Yearly Average Standard Deviation (Change)')

        for text, color in zip(new_legend.get_texts(), new_label_colors):
            text.set_color(color)
            
    plt.ylim(0, 1) 
    title = 'Combined Weighted Monthly Standard Deviation Over All Years' if weighted else 'Combined Non-Weighted Monthly Standard Deviation Over All Years'
    plt.title(title)
    plt.xlabel('Month')
    plt.xticks(rotation=90)
    plt.ylabel(std_col.replace('_', ' ').title())
    plt.tight_layout()
    combined_plot_filename = os.path.join(combined_output_folder, f"combined_{std_col}.png")
    plt.savefig(combined_plot_filename)
    plt.close()
    print(f"Combined graph saved: {combined_plot_filename}")

def calculate_percentage_change(current_distribution, previous_distribution):
    percentage_change = {}
    emotions = set(current_distribution.index).union(previous_distribution.index)
    for emotion in emotions:
        current = current_distribution.get(emotion, 0)
        previous = previous_distribution.get(emotion, 0)
        change = current - previous
        percentage_change[emotion] = change
    return percentage_change

def plot_pie_chart(data, emotion_pie_folder, year, percentage_change=None):
    print(data['Primary_Emotion'].value_counts()) 

    emotion_counts = data['Primary_Emotion'].value_counts(normalize=True) * 100
    labels = []
    label_colors = []
    for emotion, count in emotion_counts.items():
        label_text = f"{emotion}: {count:.3f}%"
        if percentage_change and emotion in percentage_change:
            change = percentage_change[emotion]
            label_text += f" ({change:+.3f}%)"
            label_colors.append('green' if change > 0 else 'red')
        else:
            label_colors.append('black')
        labels.append(label_text)

    colors = [emotion_colors.get(emotion, emotion_colors['other']) for emotion in emotion_counts.index]

    plt.figure(figsize=(10, 8))
    wedges, texts = plt.pie(emotion_counts, labels=None, colors=colors, startangle=140, counterclock=False)
    plt.axis('equal') 

    legend = plt.legend(wedges, labels, title="Emotions", loc="best")
    
    for i, text in enumerate(legend.get_texts()):
        text.set_color(label_colors[i])

    pie_chart_filename = os.path.join(emotion_pie_folder, f"{year}_emotion_pie.png")
    plt.savefig(pie_chart_filename, bbox_inches='tight')
    plt.close()

def plot_emotion_pie_charts(input_folder, emotion_pie_folder):
    previous_distribution = None
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.xlsx')])

    for file in files:
        year = file.split('.')[0]
        df = pd.read_excel(os.path.join(input_folder, file))
        
        current_distribution = df['Primary_Emotion'].value_counts(normalize=True) * 100

        if previous_distribution is not None:
            percentage_change = calculate_percentage_change(current_distribution, previous_distribution)
            plot_pie_chart(df, emotion_pie_folder, year, percentage_change)
        else:
            plot_pie_chart(df, emotion_pie_folder, year)

        previous_distribution = current_distribution


def plot_average_sentiment_over_time(input_folder, sentiment_output_folder):
    yearly_avg_sentiment_data = {}
    all_monthly_data = pd.DataFrame() 
    print("Starting average sentiment score plot generation...")
    for file in sorted(os.listdir(input_folder)):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(input_folder, file))
            df.columns = [x.lower() for x in df.columns]
            
            if 'created_utc' not in df.columns or 'sentiment_score' not in df.columns:
                print(f"Required columns are missing in the file: {file}")
                continue
            
            df['month'] = pd.to_datetime(df['created_utc']).dt.to_period('M').dt.strftime('%Y-%m')
            df['year'] = pd.to_datetime(df['created_utc']).dt.year

            monthly_avg_sentiment = df.groupby('month')['sentiment_score'].mean().reset_index()
            yearly_avg_sentiment = monthly_avg_sentiment['sentiment_score'].mean()
            year = str(df['year'].iloc[0])
            yearly_avg_sentiment_data[year] = yearly_avg_sentiment

            all_monthly_data = pd.concat([all_monthly_data, monthly_avg_sentiment.assign(year=year)])

    years = sorted(yearly_avg_sentiment_data.keys())
    yearly_changes = {}
    previous_year_sentiment = None
    for year in years:
        if previous_year_sentiment is not None:
            yearly_changes[year] = (yearly_avg_sentiment_data[year] - previous_year_sentiment) / previous_year_sentiment
        previous_year_sentiment = yearly_avg_sentiment_data[year]

    min_date = to_datetime(all_monthly_data['month']).min()

    plt.figure(figsize=(12, 7))
    for year in years:
        monthly_data = all_monthly_data[all_monthly_data['year'] == year]
        if not monthly_data.empty:
            dates = pd.to_datetime(monthly_data['month'])
            plt.plot(dates, monthly_data['sentiment_score'],
                    marker='o', linestyle='-', label=f"{year} - Avg: {yearly_avg_sentiment_data[year]:.3f}")

    plt.legend(title='Yearly Average Sentiment (Change)', loc='upper right')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().set_xlim([min_date, None]) 
    plt.xticks(rotation=90) 

    plt.title('Average Monthly Sentiment Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.ylim([-0.5, 0.5]) 
    plt.tick_params(axis='x', labelsize='small')
    plt.tight_layout()
    combined_plot_filename = os.path.join(sentiment_output_folder, "combined_average_sentiment.png")
    plt.savefig(combined_plot_filename)
    plt.close()
    print(f"Combined average sentiment score plot saved: {combined_plot_filename}")
          
plot_standard_deviation_over_time(input_folder, output_folder, weighted=True)
plot_standard_deviation_over_time(input_folder, output_folder, weighted=False)

plot_emotion_pie_charts(input_folder, emotion_pie_folder)

plot_average_sentiment_over_time(input_folder, sentiment_output_folder)