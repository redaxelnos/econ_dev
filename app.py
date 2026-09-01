import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Economic Development Gap Map", layout="wide")
st.title("Economic Development & Infrastructure Gap Analysis")

# 1. CACHE EXPENSIVE DATA CALLS
# This ensures we only download the federal/OSM data once per session
@st.cache_data
def load_census_data():
    base_url = "https://lehd.ces.census.gov/data/lodes/LODES8/pa"
    wac_21 = pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2021.csv.gz", usecols=['w_geocode', 'C000'])
    wac_16 = pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2016.csv.gz", usecols=['w_geocode', 'C000'])
    xwalk = pd.read_csv(f"{base_url}/pa_xwalk.csv.gz", usecols=['tabblk2020', 'trct'])
    
    df_jobs = pd.merge(wac_21, wac_16, on='w_geocode', suffixes=('_21', '_16'), how='outer').fillna(0)
    df_jobs['job_growth'] = df_jobs['C000_21'] - df_jobs['C000_16']
    df_jobs = pd.merge(df_jobs, xwalk, left_on='w_geocode', right_on='tabblk2020')
    tract_jobs = df_jobs.groupby('trct')['job_growth'].sum().reset_index()
    tract_jobs['trct'] = tract_jobs['trct'].astype(str)
    
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_42_tract.zip"
    gdf_tracts = gpd.read_file(tiger_url)
    gdf_mapped = gdf_tracts.merge(tract_jobs, left_on='GEOID', right_on='trct', how='left')
    gdf_mapped['job_growth'] = gdf_mapped['job_growth'].fillna(0)
    
    return gdf_mapped

@st.cache_data
def load_osm_data():
    tags = {'amenity': ['fuel', 'charging_station']}
    gdf_infra = ox.features_from_place("Pittsburgh, Pennsylvania, USA", tags=tags)
    gdf_points = gdf_infra.to_crs(epsg=3857)
    gdf_points['geometry'] = gdf_points['geometry'].centroid
    return gdf_points.to_crs(epsg=4326)

# 2. LOAD DATA
with st.spinner("Fetching Census and Infrastructure Data..."):
    gdf_mapped = load_census_data()
    gdf_infra = load_osm_data()

# 3. BUILD SIDEBAR FILTERS
st.sidebar.header("Filter Layers")

# Filter Job Growth 
min_growth = st.sidebar.slider(
    "Minimum Job Growth Threshold (2016-2021)", 
    min_value=int(gdf_mapped['job_growth'].min()), 
    max_value=int(gdf_mapped['job_growth'].max()), 
    value=0, 
    step=50
)
filtered_tracts = gdf_mapped[gdf_mapped['job_growth'] >= min_growth]

# Filter Infrastructure Types
infra_options = st.sidebar.multiselect(
    "Infrastructure Amenities",
    options=['fuel', 'charging_station'],
    default=['fuel', 'charging_station'],
    format_func=lambda x: "Gas Stations" if x == 'fuel' else "EV Chargers"
)
filtered_infra = gdf_infra[gdf_infra['amenity'].isin(infra_options)]

# 4. RENDER THE MAP
m = folium.Map(location=[40.4406, -79.9959], zoom_start=11, tiles="CartoDB positron")

# Add Choropleth
folium.Choropleth(
    geo_data=filtered_tracts,
    name='Job Creation',
    data=filtered_tracts,
    columns=['GEOID', 'job_growth'],
    key_on='feature.properties.GEOID',
    fill_color='RdYlGn', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Net Job Growth'
).add_to(m)

# Add Infrastructure Cluster
marker_cluster = MarkerCluster(name="Infrastructure").add_to(m)
for idx, row in filtered_infra.iterrows():
    amenity = row.get('amenity')
    icon_name = 'plug' if amenity == 'charging_station' else 'gas-pump'
    icon_color = 'green' if amenity == 'charging_station' else 'blue'
    
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=row.get('name', 'Infrastructure Point'),
        icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
    ).add_to(marker_cluster)

folium.LayerControl().add_to(m)

# Display in Streamlit (returned_objects=[] prevents UI lag on panning)
st_folium(m, use_container_width=True, returned_objects=[], height=600)