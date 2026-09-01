import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
import requests
import io
import numpy as np
import branca.colormap as cm
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="PA Economic Gap Analysis", layout="wide")
st.title("Pennsylvania Economic & Infrastructure Gap Analysis")

# --- SESSION STATE INITIALIZATION ---
if "selected_geoid" not in st.session_state:
    st.session_state.selected_geoid = "42003010300"  # Default Pittsburgh sample tract
if "tract_modifications" not in st.session_state:
    st.session_state.tract_modifications = {}  # Format: {geoid: [...]}

# --- REGION SELECTOR CONFIGURATION ---
REGIONS = {
    "Statewide View": {"coords": [40.9, -77.6], "zoom": 7},
    "Pittsburgh": {"coords": [40.4406, -79.9959], "zoom": 12},
    "Philadelphia": {"coords": [39.9526, -75.1652], "zoom": 12},
    "Harrisburg": {"coords": [40.2732, -76.8867], "zoom": 13},
    "Allentown": {"coords": [40.6023, -75.4714], "zoom": 13},
    "Erie": {"coords": [42.1292, -80.0851], "zoom": 13},
    "Scranton": {"coords": [41.4090, -75.6624], "zoom": 13},
    "Lancaster": {"coords": [40.0379, -76.3055], "zoom": 13}
}

selected_region = st.sidebar.selectbox("Select Target Analysis Region", list(REGIONS.keys()))
region_coords = REGIONS[selected_region]["coords"]
region_zoom = REGIONS[selected_region]["zoom"]

# 1. Federal Data (Census LEHD Job Growth)
@st.cache_data
def load_census_data():
    base_url = "https://lehd.ces.census.gov/data/lodes/LODES8/pa"
    cols = ['w_geocode', 'C000', 'CE03']
    wac_21 = pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2021.csv.gz", usecols=cols)
    wac_16 = pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2016.csv.gz", usecols=cols)
    xwalk = pd.read_csv(f"{base_url}/pa_xwalk.csv.gz", usecols=['tabblk2020', 'trct'])
    
    df_jobs = pd.merge(wac_21, wac_16, on='w_geocode', suffixes=('_21', '_16'), how='outer').fillna(0)
    df_jobs['job_growth'] = df_jobs['C000_21'] - df_jobs['C000_16']
    df_jobs['high_wage_growth'] = df_jobs['CE03_21'] - df_jobs['CE03_16']
    
    df_jobs = pd.merge(df_jobs, xwalk, left_on='w_geocode', right_on='tabblk2020')
    tract_jobs = df_jobs.groupby('trct')[['job_growth', 'high_wage_growth', 'C000_21']].sum().reset_index()
    tract_jobs['trct'] = tract_jobs['trct'].astype(str)
    
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_42_tract.zip"
    gdf_tracts = gpd.read_file(tiger_url)
    gdf_mapped = gdf_tracts.merge(tract_jobs, left_on='GEOID', right_on='trct', how='left')
    gdf_mapped['job_growth'] = gdf_mapped['job_growth'].fillna(0)
    gdf_mapped['high_wage_growth'] = gdf_mapped['high_wage_growth'].fillna(0)
    gdf_mapped['C000_21'] = gdf_mapped['C000_21'].fillna(0)
    
    np.random.seed(42)
    gdf_mapped['baseline_home_value'] = 180000 + (gdf_mapped['high_wage_growth'] * 1200) + (gdf_mapped['C000_21'] * 45)
    gdf_mapped['baseline_home_value'] = gdf_mapped['baseline_home_value'].clip(lower=95000, upper=650000)
    
    return gdf_mapped

# 2. Capital Velocity Data (WPRDC - Pittsburgh Only)
@st.cache_data
def load_permit_data():
    wprdc_url = "https://data.wprdc.org/api/3/action/datastore_search"
    params = {'resource_id': '20162fb2-7a09-43c1-b0b3-34e87754d9a8', 'q': 'COMMERCIAL', 'limit': 1000}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        response = requests.get(wprdc_url, params=params, headers=headers, timeout=10)
        df_permits = pd.DataFrame(response.json()['result']['records'])
        if df_permits.empty: return gpd.GeoDataFrame()
        df_permits['cost'] = pd.to_numeric(df_permits.get('estimated_cost', 0), errors='coerce').fillna(0)
        lat_col = 'latitude' if 'latitude' in df_permits.columns else 'y'
        lon_col = 'longitude' if 'longitude' in df_permits.columns else 'x'
        df_permits = df_permits.dropna(subset=[lat_col, lon_col])
        gdf_permits = gpd.GeoDataFrame(df_permits, geometry=gpd.points_from_xy(df_permits[lon_col], df_permits[lat_col]), crs="EPSG:4326")
        return gdf_permits[gdf_permits['cost'] > 0]
    except Exception:
        return gpd.GeoDataFrame()

# 3. OSM Economic Anchor Infrastructure Data
@st.cache_data
def load_osm_data(city_name):
    if city_name == "Statewide View": return gpd.GeoDataFrame()
    tags = {'amenity': ['bank', 'hospital', 'childcare', 'university', 'college'], 'shop': ['supermarket'], 'public_transport': ['station']}
    try:
        gdf_infra = ox.features_from_place(f"{city_name}, Pennsylvania, USA", tags=tags)
        for col in ['amenity', 'shop', 'public_transport']:
            if col not in gdf_infra.columns: gdf_infra[col] = pd.NA
        gdf_points = gdf_infra.to_crs(epsg=3857)
        gdf_points['geometry'] = gdf_points['geometry'].centroid
        return gdf_points.to_crs(epsg=4326)
    except Exception:
        return gpd.GeoDataFrame()

# 4 & 5. Federal Boundaries
@st.cache_data
def load_federal_boundaries(layer_type):
    url = "https://services6.arcgis.com/zDzo4EZXf1AjkPjO/ArcGIS/rest/services/Qualified_Census_Tracts_2025/FeatureServer/0/query" if layer_type == "QCT" else "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/Opportunity_Zones/FeatureServer/0/query"
    out_fields = "GEOID,TRACT,NAME" if layer_type == "QCT" else "TRACT,STATE_NAME"
    
    all_features = []
    offset = 0
    batch_size = 1000
    
    while True:
        params = {"where": "STATE='42' OR 1=1", "outFields": out_fields, "resultRecordCount": batch_size, "resultOffset": offset, "f": "geojson"}
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if response.ok:
                data = response.json()
                features = data.get("features", [])
                if not features: break
                all_features.extend(features)
                if len(features) < batch_size: break
                offset += batch_size
            else: break
        except Exception: break
            
    if all_features:
        geo_dict = {"type": "FeatureCollection", "features": all_features}
        gdf = gpd.GeoDataFrame.from_features(geo_dict, crs="EPSG:4326")
        if layer_type == "QCT":
            gdf['Designation'] = 'HUD Distressed Area (QCT)'
            gdf['Strategic_Note'] = 'Qualifies for LIHTC 30% basis boost & federal grants (50%+ low-income or 25%+ poverty).'
        else:
            gdf['Designation'] = 'Federal Opportunity Zone'
            gdf['Strategic_Note'] = 'Eligible for capital gains tax deferments & step-ups.'
        return gdf
    return gpd.GeoDataFrame()

# --- INITIALIZE & LOAD DATA ---
with st.spinner(f"Loading {selected_region} Data & Spatial Matrices..."):
    gdf_mapped = load_census_data()
    gdf_qct = load_federal_boundaries("QCT")
    gdf_oz = load_federal_boundaries("OZ")
    gdf_infra = load_osm_data(selected_region)
    gdf_permits = load_permit_data() if selected_region == "Pittsburgh" else gpd
