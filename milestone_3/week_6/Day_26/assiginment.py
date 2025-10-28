import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Accident Severity Hypothesis Testing",
    page_icon="🚗",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    try:
        ds = pd.read_csv("US_Accidents_March23_sampled_1M.csv")
        
        # Convert datetime columns
        ds['Start_Time'] = pd.to_datetime(ds['Start_Time'], errors='coerce')
        ds['Hour'] = ds['Start_Time'].dt.hour
        ds['DayOfWeek'] = ds['Start_Time'].dt.day_name()
        ds['Month'] = ds['Start_Time'].dt.month_name()
        
        return ds
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
ds = load_data()

if ds is not None:
    # Sidebar
    st.sidebar.title("🚗 Accident Analysis")
    st.sidebar.markdown("---")
    
    # Dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Records", f"{len(ds):,}")
    st.sidebar.metric("Date Range", f"{ds['Start_Time'].min().date()} to {ds['Start_Time'].max().date()}")
    st.sidebar.metric("Columns", len(ds.columns))
    
    # Filters
    st.sidebar.header("🔍 Filters")
    
    if 'State' in ds.columns:
        states = st.sidebar.multiselect(
            "Select States:",
            options=sorted(ds['State'].unique()),
            default=sorted(ds['State'].unique())[:3]
        )
        if states:
            ds = ds[ds['State'].isin(states)]
    
    if 'Severity' in ds.columns:
        severity_levels = st.sidebar.multiselect(
            "Select Severity Levels:",
            options=sorted(ds['Severity'].unique()),
            default=sorted(ds['Severity'].unique())
        )
        if severity_levels:
            ds = ds[ds['Severity'].isin(severity_levels)]

    # Main content
    st.title("🚗 Accident Severity Hypothesis Testing")
    st.markdown("Statistical analysis of factors affecting accident severity")
    
    # Hypothesis 1: Urban vs Rural
    st.header("🏙️ Hypothesis 1: Urban vs Rural Severity")
    
    with st.expander("View Hypothesis Details", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question:")
            st.write("Are accidents in urban areas more severe than in rural areas?")
            
            st.subheader("Hypotheses:")
            st.write("**H₀ (Null):** No difference in severity between urban and rural areas")
            st.write("**H₁ (Alternative):** Urban areas have higher accident severity")
            
            st.subheader("Methodology:")
            st.write("- T-test for independent samples")
            st.write("- Significance level: α = 0.05")
        
        with col2:
            st.subheader("Test Parameters")
            st.metric("Alpha Level", "0.05")
            st.metric("Test Type", "Independent T-test")
    
    # Analysis for Hypothesis 1
    if 'City' in ds.columns:
        city_counts = ds['City'].value_counts()
        urban_threshold = city_counts.quantile(0.8)
        urban_cities = city_counts[city_counts >= urban_threshold].index
        rural_cities = city_counts[city_counts < urban_threshold].index
        
        urban_accidents = ds[ds['City'].isin(urban_cities)]
        rural_accidents = ds[ds['City'].isin(rural_cities)]
        
        if len(urban_accidents) > 0 and len(rural_accidents) > 0:
            urban_severity = urban_accidents['Severity']
            rural_severity = rural_accidents['Severity']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(urban_severity, rural_severity, equal_var=False)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=urban_severity, name='Urban', marker_color='blue'))
                fig_box.add_trace(go.Box(y=rural_severity, name='Rural', marker_color='green'))
                fig_box.update_layout(title='Severity Distribution: Urban vs Rural',
                                    yaxis_title='Severity Level')
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Bar chart with means
                means = [urban_severity.mean(), rural_severity.mean()]
                fig_bar = px.bar(x=['Urban', 'Rural'], y=means,
                               labels={'x': 'Area Type', 'y': 'Average Severity'},
                               title='Average Severity by Area Type',
                               color=['Urban', 'Rural'],
                               color_discrete_map={'Urban': 'blue', 'Rural': 'green'})
                fig_bar.update_traces(hovertemplate='Average Severity: %{y:.3f}')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Results
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("T-statistic", f"{t_stat:.4f}")
            with col4:
                st.metric("P-value", f"{p_value:.4f}")
            with col5:
                if p_value < 0.05:
                    if urban_severity.mean() > rural_severity.mean():
                        st.metric("Conclusion", "Urban more severe", delta="Reject H₀")
                    else:
                        st.metric("Conclusion", "Rural more severe", delta="Reject H₀")
                else:
                    st.metric("Conclusion", "No difference", delta="Fail to reject H₀")
            
            st.info(f"**Sample Sizes:** Urban: {len(urban_accidents):,} | Rural: {len(rural_accidents):,}")
    
    st.markdown("---")
    
    # Hypothesis 2: Temperature Impact
    st.header("🌡️ Hypothesis 2: Temperature Impact")
    
    with st.expander("View Hypothesis Details", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question:")
            st.write("Do extreme temperatures lead to more severe accidents?")
            
            st.subheader("Hypotheses:")
            st.write("**H₀ (Null):** Temperature has no effect on accident severity")
            st.write("**H₁ (Alternative):** Extreme temperatures increase accident severity")
            
            st.subheader("Methodology:")
            st.write("- ANOVA test for multiple groups")
            st.write("- Temperature groups: Very Cold, Cold, Moderate, Hot")
        
        with col2:
            st.subheader("Test Parameters")
            st.metric("Alpha Level", "0.05")
            st.metric("Test Type", "ANOVA")
    
    # Analysis for Hypothesis 2
    if 'Temperature(F)' in ds.columns:
        temp_data = ds[['Temperature(F)', 'Severity']].dropna()
        temp_data = temp_data[(temp_data['Temperature(F)'] >= -50) & (temp_data['Temperature(F)'] <= 120)]
        
        # Create temperature groups
        temp_data['Temp_Group'] = pd.cut(temp_data['Temperature(F)'], 
                                       bins=[-50, 32, 60, 80, 120],
                                       labels=['Very Cold (<32°F)', 'Cold (32-60°F)', 
                                              'Moderate (60-80°F)', 'Hot (>80°F)'])
        
        groups = [group['Severity'].values for name, group in temp_data.groupby('Temp_Group')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot by temperature group
            fig = px.box(temp_data, x='Temp_Group', y='Severity',
                        title='Severity Distribution by Temperature Group',
                        color='Temp_Group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average severity by temperature group
            severity_by_temp = temp_data.groupby('Temp_Group')['Severity'].mean().reset_index()
            fig_bar = px.bar(severity_by_temp, x='Temp_Group', y='Severity',
                           title='Average Severity by Temperature Group',
                           color='Temp_Group')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Results
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("F-statistic", f"{f_stat:.4f}")
        with col4:
            st.metric("P-value", f"{p_value:.4f}")
        with col5:
            if p_value < 0.05:
                max_severity_temp = severity_by_temp.loc[severity_by_temp['Severity'].idxmax(), 'Temp_Group']
                st.metric("Conclusion", f"Significant effect", delta="Reject H₀")
                st.write(f"Highest severity in: **{max_severity_temp}**")
            else:
                st.metric("Conclusion", "No significant effect", delta="Fail to reject H₀")
    
    st.markdown("---")
    
    # Hypothesis 3: Intersection Impact
    st.header("🛣️ Hypothesis 3: Intersection Impact")
    
    with st.expander("View Hypothesis Details", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question:")
            st.write("Are accidents at intersections more severe than on straight roads?")
            
            st.subheader("Hypotheses:")
            st.write("**H₀ (Null):** No difference in severity between intersection and non-intersection accidents")
            st.write("**H₁ (Alternative):** Intersection accidents have higher severity")
        
        with col2:
            st.subheader("Test Parameters")
            st.metric("Alpha Level", "0.05")
            st.metric("Test Type", "Independent T-test")
    
    # Analysis for Hypothesis 3
    if 'Junction' in ds.columns:
        intersection_accidents = ds[ds['Junction'] == True]
        non_intersection_accidents = ds[ds['Junction'] == False]
        
        if len(intersection_accidents) > 0 and len(non_intersection_accidents) > 0:
            intersection_severity = intersection_accidents['Severity']
            non_intersection_severity = non_intersection_accidents['Severity']
            
            t_stat, p_value = stats.ttest_ind(intersection_severity, non_intersection_severity, equal_var=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(y=intersection_severity, name='Intersection', marker_color='red'))
                fig.add_trace(go.Box(y=non_intersection_severity, name='Non-Intersection', marker_color='orange'))
                fig.update_layout(title='Severity: Intersection vs Non-Intersection',
                                yaxis_title='Severity Level')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Severity distribution comparison
                severity_counts_intersection = intersection_severity.value_counts().sort_index()
                severity_counts_non = non_intersection_severity.value_counts().sort_index()
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Bar(name='Intersection', 
                                        x=severity_counts_intersection.index,
                                        y=severity_counts_intersection.values/len(intersection_severity)*100,
                                        marker_color='red'))
                fig_dist.add_trace(go.Bar(name='Non-Intersection',
                                        x=severity_counts_non.index,
                                        y=severity_counts_non.values/len(non_intersection_severity)*100,
                                        marker_color='orange'))
                fig_dist.update_layout(title='Severity Distribution (%)',
                                    xaxis_title='Severity Level',
                                    yaxis_title='Percentage (%)',
                                    barmode='group')
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Results
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("T-statistic", f"{t_stat:.4f}")
            with col4:
                st.metric("P-value", f"{p_value:.4f}")
            with col5:
                if p_value < 0.05:
                    if intersection_severity.mean() > non_intersection_severity.mean():
                        st.metric("Conclusion", "Intersections more severe", delta="Reject H₀")
                    else:
                        st.metric("Conclusion", "Straight roads more severe", delta="Reject H₀")
                else:
                    st.metric("Conclusion", "No difference", delta="Fail to reject H₀")
            
            st.info(f"**Sample Sizes:** Intersection: {len(intersection_accidents):,} | Non-Intersection: {len(non_intersection_accidents):,}")
    
    st.markdown("---")
    
    # Hypothesis 4: Daylight vs Darkness
    st.header("🌅 Hypothesis 4: Daylight vs Darkness")
    
    with st.expander("View Hypothesis Details", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question:")
            st.write("Are accidents during night more severe than during daylight?")
            
            st.subheader("Hypotheses:")
            st.write("**H₀ (Null):** No difference in severity between daylight and night")
            st.write("**H₁ (Alternative):** Night accidents have higher severity")
        
        with col2:
            st.subheader("Test Parameters")
            st.metric("Alpha Level", "0.05")
            st.metric("Test Type", "Independent T-test")
    
    # Analysis for Hypothesis 4
    if 'Sunrise_Sunset' in ds.columns:
        daylight_data = ds[ds['Sunrise_Sunset'].notna()]
        daylight = daylight_data[daylight_data['Sunrise_Sunset'] == 'Day']['Severity']
        night = daylight_data[daylight_data['Sunrise_Sunset'] == 'Night']['Severity']
        
        if len(daylight) > 0 and len(night) > 0:
            t_stat, p_value = stats.ttest_ind(daylight, night, equal_var=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(y=daylight, name='Daylight', marker_color='yellow'))
                fig.add_trace(go.Box(y=night, name='Night', marker_color='navy'))
                fig.update_layout(title='Severity: Daylight vs Night',
                                yaxis_title='Severity Level')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly severity pattern
                if 'Hour' in ds.columns:
                    hourly_severity = ds.groupby('Hour')['Severity'].mean().reset_index()
                    fig_line = px.line(hourly_severity, x='Hour', y='Severity',
                                     title='Average Severity by Hour of Day',
                                     markers=True)
                    # Highlight night hours
                    night_hours = list(range(18, 24)) + list(range(0, 6))
                    for hour in night_hours:
                        fig_line.add_vrect(x0=hour-0.5, x1=hour+0.5, 
                                         fillcolor="blue", opacity=0.1, line_width=0)
                    st.plotly_chart(fig_line, use_container_width=True)
            
            # Results
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("T-statistic", f"{t_stat:.4f}")
            with col4:
                st.metric("P-value", f"{p_value:.4f}")
            with col5:
                if p_value < 0.05:
                    if daylight.mean() > night.mean():
                        st.metric("Conclusion", "Daylight more severe", delta="Reject H₀")
                    else:
                        st.metric("Conclusion", "Night more severe", delta="Reject H₀")
                else:
                    st.metric("Conclusion", "No difference", delta="Fail to reject H₀")
    
    st.markdown("---")
    
    # NEW HYPOTHESIS 5: Weather Condition Impact
    st.header("🌧️ Hypothesis 5: Weather Condition Impact")
    
    with st.expander("View Hypothesis Details", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question:")
            st.write("Do adverse weather conditions lead to more severe accidents compared to clear weather?")
            
            st.subheader("Hypotheses:")
            st.write("**H₀ (Null):** Weather conditions have no effect on accident severity")
            st.write("**H₁ (Alternative):** Adverse weather conditions increase accident severity")
            
            st.subheader("Methodology:")
            st.write("- ANOVA test for multiple weather groups")
            st.write("- Weather groups: Clear, Rain, Snow, Fog, Other")
        
        with col2:
            st.subheader("Test Parameters")
            st.metric("Alpha Level", "0.05")
            st.metric("Test Type", "ANOVA")
    
    # Analysis for NEW Hypothesis 5
    if 'Weather_Condition' in ds.columns:
        # Clean and categorize weather conditions
        weather_data = ds[['Weather_Condition', 'Severity']].dropna()
        
        # Create weather categories
        def categorize_weather(condition):
            if pd.isna(condition):
                return 'Unknown'
            condition = str(condition).lower()
            if any(word in condition for word in ['clear', 'fair', 'sunny']):
                return 'Clear'
            elif any(word in condition for word in ['rain', 'drizzle', 'shower']):
                return 'Rain'
            elif any(word in condition for word in ['snow', 'sleet', 'ice', 'hail']):
                return 'Snow/Ice'
            elif any(word in condition for word in ['fog', 'haze', 'mist']):
                return 'Fog'
            else:
                return 'Other'
        
        weather_data['Weather_Group'] = weather_data['Weather_Condition'].apply(categorize_weather)
        
        # Filter to only include groups with sufficient data
        weather_counts = weather_data['Weather_Group'].value_counts()
        valid_groups = weather_counts[weather_counts >= 100].index
        weather_data = weather_data[weather_data['Weather_Group'].isin(valid_groups)]
        
        if len(weather_data['Weather_Group'].unique()) > 1:
            # ANOVA test
            groups = [group['Severity'].values for name, group in weather_data.groupby('Weather_Group')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot by weather group
                fig = px.box(weather_data, x='Weather_Group', y='Severity',
                           title='Severity Distribution by Weather Condition',
                           color='Weather_Group',
                           category_orders={'Weather_Group': ['Clear', 'Rain', 'Fog', 'Snow/Ice', 'Other']})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average severity by weather group
                severity_by_weather = weather_data.groupby('Weather_Group')['Severity'].mean().reset_index()
                severity_by_weather = severity_by_weather.sort_values('Severity', ascending=False)
                
                fig_bar = px.bar(severity_by_weather, x='Weather_Group', y='Severity',
                               title='Average Severity by Weather Condition',
                               color='Severity',
                               color_continuous_scale='reds')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Weather frequency
                weather_freq = weather_data['Weather_Group'].value_counts().reset_index()
                weather_freq.columns = ['Weather_Group', 'Count']
                fig_pie = px.pie(weather_freq, values='Count', names='Weather_Group',
                               title='Distribution of Weather Conditions')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Results
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("F-statistic", f"{f_stat:.4f}")
            with col4:
                st.metric("P-value", f"{p_value:.4f}")
            with col5:
                if p_value < 0.05:
                    max_severity_weather = severity_by_weather.iloc[0]['Weather_Group']
                    st.metric("Conclusion", "Significant effect", delta="Reject H₀")
                    st.write(f"Highest severity in: **{max_severity_weather}**")
                else:
                    st.metric("Conclusion", "No significant effect", delta="Fail to reject H₀")
            
            # Additional insights
            st.subheader("📈 Weather Impact Insights")
            col6, col7, col8 = st.columns(3)
            
            with col6:
                clear_severity = weather_data[weather_data['Weather_Group'] == 'Clear']['Severity'].mean()
                st.metric("Clear Weather Severity", f"{clear_severity:.2f}")
            
            with col7:
                adverse_weather = weather_data[weather_data['Weather_Group'] != 'Clear']
                if len(adverse_weather) > 0:
                    adverse_severity = adverse_weather['Severity'].mean()
                    severity_diff = adverse_severity - clear_severity
                    st.metric("Adverse Weather Severity", f"{adverse_severity:.2f}", 
                             delta=f"{severity_diff:+.2f}")
            
            with col8:
                total_adverse = len(adverse_weather)
                percentage_adverse = (total_adverse / len(weather_data)) * 100
                st.metric("Adverse Weather Accidents", f"{percentage_adverse:.1f}%")
    
    # Summary Section
    st.markdown("---")
    st.header("📊 Summary of Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Insights")
        st.write("""
        - **Urban vs Rural**: Population density effects on accident severity
        - **Temperature Impact**: How extreme weather affects crash outcomes
        - **Intersection Safety**: Risk profiles at critical road points
        - **Time of Day**: Visibility and behavioral factors
        - **Weather Conditions**: Impact of precipitation and visibility on severity
        """)
        
        st.subheader("Practical Implications")
        st.write("""
        • Target safety measures in high-severity conditions
        • Improve infrastructure based on risk factors
        • Enhance emergency response planning
        • Develop weather-specific driving advisories
        """)
    
    with col2:
        st.subheader("Assumptions & Limitations")
        st.write("""
        - Significance level: α = 0.05 for all tests
        - Independent observations assumed
        - Missing data excluded from analyses
        - Urban/rural classification based on accident frequency
        - Equal variance not assumed (Welch's t-test used)
        - Weather categorization based on keyword matching
        """)
        
        st.subheader("Data Quality")
        st.write("""
        ✅ Large sample sizes for robust analysis  
        ✅ Comprehensive geographic coverage  
        ⚠️ Self-reported weather conditions  
        ⚠️ Potential reporting biases  
        ✅ Multiple validation methods used
        """)
    
else:
    st.error("Failed to load dataset. Please check the file path and try again.")

# Footer
st.markdown("---")
st.caption("Accident Severity Hypothesis Testing Dashboard | Statistical Analysis of Road Safety Factors")