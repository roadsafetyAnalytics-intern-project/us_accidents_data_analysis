import pandas as pd
from sklearn.linear_model import LinearRegression

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "C:/Users/win10/Desktop/Project_Aug25/US_Accidents_March23.csv"
OUTPUT_PATH = "US_Accidents_preprocessed.csv"

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Initial shape: {df.shape}")

# ============================================================================
# STEP 2: REMOVE DUPLICATES
# ============================================================================
print("\nRemoving duplicates...")
df = df.drop_duplicates(subset="ID")
print(f"Shape after deduplication: {df.shape}")

# ============================================================================
# STEP 3: DROP HIGH MISSINGNESS COLUMNS (>30%)
# ============================================================================
print("\nDropping columns with >30% missing values...")
missing_percent = round((df.isnull().sum() / df.shape[0]) * 100, 2)
remove_cols = missing_percent[missing_percent > 30].index.tolist()
print(f"Columns to remove: {remove_cols}")
df.drop(columns=remove_cols, inplace=True)

# ============================================================================
# STEP 4: DROP NON-ANALYTICAL COLUMNS
# ============================================================================
print("\nDropping non-analytical columns...")
drop_cols = ["ID", "Source", "Description", "Street", "Country", 
             "Zipcode", "Timezone", "Airport_Code", "Amenity"]
drop_cols_existing = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols_existing)
print(f"Shape after column removal: {df.shape}")

# ============================================================================
# STEP 5: PARSE AND VALIDATE TEMPORAL DATA
# ============================================================================
print("\nParsing temporal data...")
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
df = df.dropna(subset=["Start_Time", "End_Time"])
print(f"Shape after temporal validation: {df.shape}")

# ============================================================================
# STEP 6: VALIDATE GEOGRAPHIC DATA
# ============================================================================
print("\nValidating geographic data...")
df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
df = df.dropna(subset=["Start_Lat", "Start_Lng"])
df.rename(columns={'Start_Lat': 'Latitude', 'Start_Lng': 'Longitude'}, inplace=True)
print(f"Shape after geographic validation: {df.shape}")

# ============================================================================
# STEP 7: FILTER SEVERITY CLASSES
# ============================================================================
print("\nFiltering severity classes...")
df = df[df["Severity"].isin([1, 2, 3, 4])]
print(f"Shape after severity filtering: {df.shape}")

# ============================================================================
# STEP 8: DROP ROWS WITH LOW MISSINGNESS (<3%)
# ============================================================================
print("\nDropping rows with low missingness...")
missing_percent = (df.isnull().sum() / df.shape[0]) * 100
low_missing_cols = missing_percent[(missing_percent > 0) & (missing_percent <= 3)].index.tolist()
if low_missing_cols:
    print(f"Columns with 0-3% missing: {low_missing_cols}")
    df.dropna(subset=low_missing_cols, inplace=True)
    print(f"Shape after dropping low-missing rows: {df.shape}")

# ============================================================================
# STEP 9: TARGETED WEATHER IMPUTATION
# ============================================================================
print("\nPerforming targeted imputation...")

# Wind Speed - Median imputation
if 'Wind_Speed(mph)' in df.columns and df['Wind_Speed(mph)'].isnull().any():
    wind_median = df['Wind_Speed(mph)'].median()
    df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(wind_median)
    print(f"Imputed Wind_Speed(mph) with median: {wind_median:.2f}")

# Precipitation - Zero-fill (no rain assumption)
if 'Precipitation(in)' in df.columns and df['Precipitation(in)'].isnull().any():
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0.0)
    print("Imputed Precipitation(in) with zero-fill")

# Wind Chill - Regression-based imputation
if 'Wind_Chill(F)' in df.columns and df['Wind_Chill(F)'].isnull().any():
    reg_features = ['Wind_Speed(mph)', 'Temperature(F)', 'Humidity(%)']
    if all(col in df.columns for col in reg_features):
        known_wc = df[df['Wind_Chill(F)'].notna()]
        unknown_wc = df[df['Wind_Chill(F)'].isna()]
        if len(unknown_wc) > 0:
            X_train = known_wc[reg_features]
            y_train = known_wc['Wind_Chill(F)']
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            X_pred = unknown_wc[reg_features]
            predicted_wc = reg.predict(X_pred)
            df.loc[df['Wind_Chill(F)'].isna(), 'Wind_Chill(F)'] = predicted_wc
            print(f"Imputed Wind_Chill(F) using regression ({len(unknown_wc)} values)")

# ============================================================================
# STEP 10: GENERAL NUMERIC IMPUTATION
# ============================================================================
print("\nGeneral numeric imputation...")
num_cols = df.select_dtypes(include="number").columns.tolist()
imputed_cols = []
for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
        imputed_cols.append(col)
if imputed_cols:
    print(f"Imputed {len(imputed_cols)} columns with median: {imputed_cols}")

# ============================================================================
# STEP 11: FEATURE ENGINEERING - TEMPORAL
# ============================================================================
print("\nCreating temporal features...")
df["Duration_Minutes"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
df['Year'] = df["Start_Time"].dt.year
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.weekday
df["Month"] = df["Start_Time"].dt.month
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
print("Created: Duration_Minutes, Year, Hour, DayOfWeek, Month, IsWeekend")

# ============================================================================
# STEP 12: FEATURE ENGINEERING - CATEGORICAL
# ============================================================================
print("\nEncoding categorical features...")
bool_cols = ["Roundabout", "Station", "Stop", "Traffic_Calming", 
             "Traffic_Signal", "Turning_Loop"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

if "Sunrise_Sunset" in df.columns:
    df["IsDay"] = (df["Sunrise_Sunset"] == "Day").astype(int)
    print("Encoded: Boolean traffic features + IsDay")

# ============================================================================
# STEP 13: DROP REDUNDANT FEATURES
# ============================================================================
print("\nDropping redundant features...")
redundant_cols = ["Start_Time", "End_Time", "Weather_Timestamp", 
                  "Civil_Twilight", "Nautical_Twilight", 
                  "Astronomical_Twilight", "Sunrise_Sunset"]
redundant_cols_existing = [col for col in redundant_cols if col in df.columns]
df = df.drop(columns=redundant_cols_existing)
print(f"Shape after dropping redundant features: {df.shape}")

# ============================================================================
# STEP 14: FINAL CLEANUP
# ============================================================================
print("\nFinal cleanup...")
rows_before = len(df)
df = df.dropna()
rows_dropped = rows_before - len(df)
print(f"Dropped {rows_dropped} rows with remaining NaN values")

# ============================================================================
# STEP 15: SAVE PREPROCESSED DATA
# ============================================================================
print(f"\nSaving preprocessed data to {OUTPUT_PATH}...")
df.to_csv(OUTPUT_PATH, index=False)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"Final shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
print(f"\nData saved to: {OUTPUT_PATH}")
print("="*80)
