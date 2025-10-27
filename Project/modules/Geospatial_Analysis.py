import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
import numpy as np

# State abbreviation to full name mapping for UI clarity
us_state_abbrev = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia"
}

def run():
    st.header("Geospatial Accident Analysis with Hotspot Counts")

    df = pd.read_csv("data/US_Accidents_preprocessed.csv")
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

    geog_level = st.radio(
        "Select geography level",
        ["Country", "State", "City"],
        index=0
    )

    vis_type = st.radio(
        "Select visualization type",
        ["Point Map", "Hotspot Density"],
        index=0
    )

    severity_options = sorted(df["Severity"].unique())
    selected_severity = st.selectbox(
        "Select Severity Level",
        options=[''] + [str(s) for s in severity_options],
        index=0
    )

    if selected_severity == '':
        st.info("Please select a severity level to display data.")
        return

    selected_severity_value = int(selected_severity)
    filtered_df = df.copy()
    region_label = None
    zoom = 3
    center = dict(lat=39, lon=-98)  # default USA center

    if geog_level == "Country":
        region_label = "Country" if "Country" in df.columns else None
        filtered_df = filtered_df[filtered_df["Severity"] == selected_severity_value]

    elif geog_level == "State":
        region_label = "State"
        unique_state_abbrevs = sorted(filtered_df["State"].dropna().unique())
        state_fullnames = [us_state_abbrev.get(abbr, abbr) for abbr in unique_state_abbrevs]
        state_name_to_abbrev = {full: abbr for full, abbr in zip(state_fullnames, unique_state_abbrevs)}

        selected_state_name = st.selectbox("Select State", options=[''] + state_fullnames)

        if selected_state_name == '':
            st.info("Please select a state to display data.")
            return
        selected_state_abbr = state_name_to_abbrev[selected_state_name]

        filtered_df = filtered_df[filtered_df["State"] == selected_state_abbr]
        filtered_df = filtered_df[filtered_df["Severity"] == selected_severity_value]

        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        center = dict(lat=center_lat, lon=center_lon)
        zoom = 6

    elif geog_level == "City":
        region_label = "City"
        city_options = sorted(filtered_df["City"].dropna().unique())
        selected_city = st.selectbox("Select City", options=[''] + city_options)

        if selected_city == '':
            st.info("Please select a city to display data.")
            return

        filtered_df = filtered_df[filtered_df["City"] == selected_city]
        filtered_df = filtered_df[filtered_df["Severity"] == selected_severity_value]

        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        center = dict(lat=center_lat, lon=center_lon)
        zoom = 9

    severity_color_map = {
        1: "green",
        2: "yellow",
        3: "orange",
        4: "red"
    }

    if filtered_df.empty:
        st.info("No accidents found for selected criteria.")
        return

    if vis_type == "Point Map":
        hover_data = {region_label: True} if region_label else {}
        fig = px.scatter_mapbox(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color=filtered_df["Severity"].astype(str),
            color_discrete_map={str(k): v for k, v in severity_color_map.items()},
            zoom=zoom,
            center=center,
            mapbox_style="carto-positron",
            hover_data=hover_data
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    else:
        coords = filtered_df[['latitude', 'longitude']].to_numpy()
        radians_coords = np.radians(coords)
        kms_per_radian = 6371.0088
        epsilon = 1.0 / kms_per_radian  # 1 km radius for clustering
        db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
        cluster_labels = db.fit_predict(radians_coords)
        filtered_df = filtered_df.assign(cluster=cluster_labels)

        clusters = filtered_df[filtered_df['cluster'] != -1]
        if clusters.empty:
            st.info("No hotspots detected for the selected criteria.")
            return

        cluster_agg = clusters.groupby('cluster').agg(
            accident_count=('cluster', 'count'),
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean')
        ).reset_index()
        cluster_agg['Severity'] = selected_severity_value

        fig = px.scatter_mapbox(
            cluster_agg,
            lat='latitude',
            lon='longitude',
            size='accident_count',
            color=cluster_agg["Severity"].astype(str),
            color_discrete_map={str(k): v for k, v in severity_color_map.items()},
            size_max=30,
            zoom=zoom,
            center=center,
            mapbox_style="carto-positron",
            hover_name='Severity',
            hover_data={
                "accident_count": True,
                "latitude": ':.4f',
                "longitude": ':.4f'
            },
            title="Accident Hotspots with Clustered Counts"
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Point size corresponds to accident count at each hotspot cluster.")
