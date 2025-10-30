import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    st.title("US RoadSafe Analytics")
    st.write("Analyze and visualize U.S. road accident trends to improve road safety awareness.")
    
    # Load full dataset directly
    df = pd.read_csv("data/US_Accidents_March23.csv")
    st.info("Loaded full dataset. This may take longer.")
    
    # Display key metrics
    total_accidents = len(df)
    avg_severity = round(df['Severity'].mean(), 2)
    st.metric("Total Accidents", total_accidents)
    st.metric("Average Severity", avg_severity)

    # Show severity distribution histogram using Plotly for interactivity
    fig = px.histogram(df, 
                       x="Severity", 
                       nbins=4, 
                       title="Severity Distribution",
                       labels={"Severity": "Accident Severity"},
                       color_discrete_sequence=["#EF553B"])
    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis_title="Count",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis: severity counts
    severity_counts = df['Severity'].value_counts().sort_index()
    st.write("### Severity Counts")
    st.bar_chart(severity_counts)

    # Allow user to select severity threshold to filter data
    min_severity = st.slider("Filter accidents with minimum severity:", min_value=1, max_value=4, value=1)
    filtered_df = df[df['Severity'] >= min_severity]
    st.write(f"Showing {len(filtered_df):,} accidents with severity >= {min_severity}")

    st.dataframe(filtered_df.head(10), use_container_width=True)
