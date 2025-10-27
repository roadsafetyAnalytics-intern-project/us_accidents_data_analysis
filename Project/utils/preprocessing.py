import pandas as pd

def clean_data(df):
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df = df.dropna(subset=['Weather_Condition', 'Visibility(mi)', 'Severity'])
    df['Hour'] = df['Start_Time'].dt.hour
    df['Month'] = df['Start_Time'].dt.month
    df['Weekday'] = df['Start_Time'].dt.day_name()
    return df
