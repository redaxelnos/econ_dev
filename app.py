import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
import requests
import io
import numpy as np
import branca.colormap as cm
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="PA Economic Gap Analysis", layout="wide")
st.title("Pennsylvania Economic & Infrastructure Gap Analysis")

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
    tract_jobs = df_jobs.groupby('trct')[['job_growth', 'high_wage_growth']].sum().reset_index()
    tract_jobs['trct'] = tract_jobs['trct'].astype(str)
    
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_42_tract.zip"
    gdf_tracts = gpd.read_file(tiger_url)
    gdf_mapped = gdf_tracts.merge(tract_jobs, left_on='GEOID', right_on='trct', how='left')
    gdf_mapped['job_growth'] = gdf_mapped['job_growth'].fillna(0)
    gdf_mapped['high_wage_growth'] = gdf_mapped['high_wage_growth'].fillna(0)
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

# 4 & 5. Federal Boundaries with Paginator
@st.cache_data
def load_federal_boundaries(layer_type):
    url = "https://services6.arcgis.com/zDzo4EZXf1AjkPjO/ArcGIS/rest/services/Qualified_Census_Tracts_2025/FeatureServer/0/query" if layer_type == "QCT" else "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/Opportunity_Zones/FeatureServer/0/query"
    out_fields = "GEOID,TRACT,NAME" if layer_type == "QCT" else "TRACT,STATE_NAME"
    
    all_features = []
    offset = 0
    batch_size = 1000
    
    while True:
        params = {
            "where": "STATE='42' OR 1=1",
            "outFields": out_fields,
            "resultRecordCount": batch_size,
            "resultOffset": offset,
            "f": "geojson"
        }
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
with st.spinner(f"Loading {selected_region} Data..."):
    gdf_mapped = load_census_data()
    gdf_qct = load_federal_boundaries("QCT")
    gdf_oz = load_federal_boundaries("OZ")
    gdf_infra = load_osm_data(selected_region)
    gdf_permits = load_permit_data() if selected_region == "Pittsburgh" else gpd.GeoDataFrame()

# --- PRECISE DIAGNOSTIC EVALUATION ---
if not gdf_qct.empty and 'GEOID' in gdf_qct.columns:
    qct_ids = set(gdf_qct['GEOID'].astype(str))
else:
    qct_ids = set()

if not gdf_oz.empty and 'TRACT' in gdf_oz.columns:
    oz_ids = set(gdf_oz['TRACT'].astype(str))
else:
    oz_ids = set()

high_wage_threshold = np.percentile(gdf_mapped['high_wage_growth'].dropna(), 75)

def evaluate_investment_risk(row):
    geoid = str(row['GEOID'])
    is_distressed = (geoid in qct_ids) or (geoid in oz_ids) or any(geoid.endswith(t) for t in oz_ids)
    growth = row['job_growth']
    high_wage = row['high_wage_growth']
    
    job_str = f"Net job change: {int(growth):+d}"
    hw_str = f"High-Wage change: {int(high_wage):+d} (Threshold: +{int(high_wage_threshold)})"
    
    if growth <= -30:
        return f"⚠️ Severely Disadvantaged / Critical Contraction | {job_str} | {hw_str} (Severe Job Loss)."
    elif is_distressed:
        if growth < -10:
            return f"🔴 High-Risk / Caution (Distressed + Decline) | {job_str} | {hw_str}."
        elif growth < 20:
            return f"🟡 Distressed / Stagnant (Needs Catalyst) | {job_str} | {hw_str}."
        else:
            return f"🟢 Distressed / High-Growth Opportunity | {job_str} | {hw_str}."
    else:
        if high_wage >= high_wage_threshold and growth > 10:
            return f"🌟 High Opportunity Growth Hub | {job_str} | {hw_str} (Top-Tier Expansion)."
        elif growth < -10:
            return f"⚠️ Declining Standard Tract | {job_str} | {hw_str} (Severe contraction outside distressed bounds)."
        else:
            return f"⚪ Stable / Moderate Growth | {job_str} | {hw_str}."

gdf_mapped['Investment_Rating'] = gdf_mapped.apply(evaluate_investment_risk, axis=1)

gdf_high_opp = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High Opportunity", na=False)].copy()
gdf_high_risk = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High-Risk", na=False)].copy()
gdf_sev_disadv = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("Severely Disadvantaged", na=False)].copy()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("Analysis Focus Mode")
analysis_mode = st.sidebar.radio(
    "Select Map Objective",
    [
        "Full Spectrum View (All Tracts)", 
        "⚠️ Severely Disadvantaged & High-Need Focus (Critical Intervention)",
        "🚨 Turnaround & Intervention Target Focus (Declining/Distressed Only)",
        "🌟 High-Growth Scaling Focus (Expansion Hubs Only)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Economic Vitality Layers")
with st.sidebar.expander("ℹ️ Understanding LEHD Data (WAC)", expanded=False):
    st.markdown("""
    - **LEHD (Longitudinal Employer-Household Dynamics):** Tracks jobs where people work (workplace blocks), not where they live.
    - **Total Job Growth (All Wages):** Net change in all jobs combined (2016-2021). Good for broad economic activity, but can mask low-wage churn replacing high-wage jobs.
    - **High-Wage Job Growth (Exceeding $40k/yr):** Isolates jobs earning greater than $3,333/month (CE03 tier). True indicator of regional wealth creation and sustainable commercial health.
    """)

base_metric = st.sidebar.radio("Base Heatmap Metric (LEHD)", ["Total Job Growth (All Wages)", "High-Wage Job Growth (Exceeding $40k/yr)"])
metric_col = 'job_growth' if base_metric == "Total Job Growth (All Wages)" else 'high_wage_growth'

# Smart Defaults & Filtering based on Analysis Focus Mode with Layer Synchronization
if analysis_mode == "⚠️ Severely Disadvantaged & High-Need Focus (Critical Intervention)":
    filtered_tracts = gdf_mapped[gdf_mapped['job_growth'] <= -30]
    st.sidebar.error("Showing *only* severely disadvantaged tracts experiencing severe job contraction (net loss of 30+ jobs).")
    default_high_opp = False
    default_high_risk = True
    default_qct = True
    default_oz = True
elif analysis_mode == "🚨 Turnaround & Intervention Target Focus (Declining/Distressed Only)":
    filtered_tracts = gdf_mapped[
        (gdf_mapped[metric_col] < 20) & 
        (gdf_mapped['Investment_Rating'].str.contains("High-Risk|Distressed", na=False))
    ]
    st.sidebar.warning("Showing *only* distressed or declining tracts requiring capital intervention.")
    default_high_opp = False
    default_high_risk = True
    default_qct = True
    default_oz = True
elif analysis_mode == "🌟 High-Growth Scaling Focus (Expansion Hubs Only)":
    filtered_tracts = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High Opportunity|High-Growth", na=False)]
    st.sidebar.success("Showing *only* high-growth expansion and opportunity hubs.")
    default_high_opp = True
    default_high_risk = False
    default_qct = False
    default_oz = False
else:
    min_growth = st.sidebar.slider(
        "Minimum Job Growth Threshold", 
        min_value=int(gdf_mapped[metric_col].min()), 
        max_value=int(gdf_mapped[metric_col].max()), 
        value=int(gdf_mapped[metric_col].min()), 
        step=50
    )
    filtered_tracts = gdf_mapped[gdf_mapped[metric_col] >= min_growth]
    default_high_opp = True
    default_high_risk = True
    default_qct = True
    default_oz = True

if selected_region == "Pittsburgh":
    show_permits = st.sidebar.checkbox("Overlay Capital Investment Heatmap (WPRDC)", value=True)
else:
    show_permits = False

st.sidebar.markdown("---")
st.sidebar.header("Policy & Opportunity Boundaries")
with st.sidebar.expander("ℹ️ Complete Statutory & Algorithmic Makeup", expanded=False):
    st.markdown("""
    - **Severely Disadvantaged Zones (Crimson):** Tracts experiencing severe job hemorrhage (loss of 30+ jobs). Requires emergency structural intervention.
    - **High Opportunity Hubs (Neon Yellow):** Non-distressed tracts where high-wage expansion exceeds the regional 75th percentile. Framed for private investment scaling.
    - **High-Risk / Caution Zones (Neon Red):** Distressed tracts experiencing net job decline (less than -10 jobs). Flags structural headwinds where tax incentives alone historically fail.
    - **HUD QCT (Neon Blue):** Statutory low-income tracts (50%+ households under 60% AMGI or 25%+ poverty). Unlocks LIHTC 30% basis boosts.
    - **Opportunity Zones (Neon White):** Treasury-certified low-income communities designated for capital gains tax deferment benefits.
    """)

show_high_opp = st.sidebar.checkbox("High Opportunity Hubs - Neon Yellow Infill", value=default_high_opp)
show_high_risk = st.sidebar.checkbox("High-Risk / Caution Zones - Neon Red Infill", value=default_high_risk)
show_qct = st.sidebar.checkbox("Distressed Areas (HUD QCT) - Neon Blue Infill", value=default_qct)
show_oz = st.sidebar.checkbox("Opportunity Zones (OZ) - Neon White Infill", value=default_oz)

st.sidebar.markdown("---")
st.sidebar.header("Economic Anchors")
if selected_region == "Statewide View":
    st.sidebar.warning("Select a specific region at the top to view local economic anchors.")
    filtered_infra = gpd.GeoDataFrame()
else:
    infra_options = st.sidebar.multiselect(
        "Infrastructure & Amenities",
        options=['bank', 'hospital', 'childcare', 'university', 'college', 'supermarket', 'station'],
        default=['bank', 'hospital', 'supermarket', 'childcare'],
        format_func=lambda x: x.title() if x not in ['supermarket', 'station'] else ('Grocery Stores' if x == 'supermarket' else 'Transit Stations')
    )
    if not gdf_infra.empty:
        mask = (gdf_infra.get('amenity', pd.Series()).isin(infra_options) | gdf_infra.get('shop', pd.Series()).isin(infra_options) | gdf_infra.get('public_transport', pd.Series()).isin(infra_options))
        filtered_infra = gdf_infra[mask]
    else:
        filtered_infra = gpd.GeoDataFrame()

# --- RENDER MAP ---
m = folium.Map(location=region_coords, zoom_start=region_zoom, tiles="OpenStreetMap")

# Layer 1: Absolute Global Percentile-Clamped Base Map
if not filtered_tracts.empty:
    global_p5, global_p95 = np.percentile(gdf_mapped[metric_col].dropna(), [5, 95])
    vmin = min(global_p5, gdf_mapped[metric_col].min())
    vmax = max(global_p95, gdf_mapped[metric_col].max())
    if vmin == vmax: vmax += 1

    colormap = cm.LinearColormap(colors=['#d73027', '#fee08b', '#1a9850'], vmin=vmin, vmax=vmax)
    colormap.caption = f'Net Growth: {base_metric} (Absolute Statewide Scale)'
    colormap.add_to(m)

    folium.GeoJson(
        filtered_tracts,
        name='Job Creation Base',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties'][metric_col]) if feature['properties'][metric_col] is not None else 'transparent',
            'color': '#333333',
            'weight': 0.4,
            'fillOpacity': 0.75,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['GEOID', 'job_growth', 'high_wage_growth', 'Investment_Rating'],
            aliases=['Census Tract ID:', 'Total Job Growth:', 'High-Wage Growth (Exceeding $40k):', 'Detailed Diagnostic Evaluation:'],
            localize=True,
            sticky=True,
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"
        )
    ).add_to(m)

# Layer 2: Permit Capital Density Heatmap
if show_permits and not gdf_permits.empty:
    heat_data = [[row.geometry.y, row.geometry.x, row['cost']] for idx, row in gdf_permits.iterrows()]
    HeatMap(heat_data, name="Capital Investment Density", radius=15, blur=10, max_zoom=1).add_to(m)

# Layer 3: High Opportunity Hubs (Neon Yellow Infill & Border)
if show_high_opp and not gdf_high_opp.empty:
    folium.GeoJson(
        gdf_high_opp,
        name="High Opportunity Growth Hubs",
        style_function=lambda x: {
            'color': '#FFFF00',
            'weight': 3.0,
            'fillColor': '#FFFF00',
            'fillOpacity': 0.3,
            'dashArray': '2, 2'
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['GEOID', 'Investment_Rating', 'high_wage_growth'],
            aliases=['Tract ID:', 'Classification:', 'High-Wage Net Gain:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal;"
        )
    ).add_to(m)

# Layer 4: High-Risk / Caution Zones (Neon Red Infill & Border)
if show_high_risk and not gdf_high_risk.empty:
    folium.GeoJson(
        gdf_high_risk,
        name="High-Risk / Caution Zones",
        style_function=lambda x: {
            'color': '#FF0055',
            'weight': 3.5,
            'fillColor': '#FF0055',
            'fillOpacity': 0.3,
            'dashArray': '5, 3'
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['GEOID', 'Investment_Rating', 'job_growth'],
            aliases=['Tract ID:', 'Risk Flag:', 'Net Job Loss:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal;"
        )
    ).add_to(m)

# Layer 5: HUD QCT (Neon Blue Infill & Border)
if show_qct and not gdf_qct.empty:
    folium.GeoJson(
        gdf_qct, 
        name="HUD Qualified Census Tracts (QCT)", 
        style_function=lambda x: {
            'color': '#00FFFF',
            'weight': 3.5, 
            'fillColor': '#00FFFF',
            'fillOpacity': 0.25,
            'dashArray': '4, 4'
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['GEOID', 'Designation', 'Strategic_Note'],
            aliases=['QCT Tract ID:', 'Boundary Type:', 'Policy Objective:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal;"
        )
    ).add_to(m)

# Layer 6: Opportunity Zones (Neon White Infill & Bright Border)
if show_oz and not gdf_oz.empty:
    folium.GeoJson(
        gdf_oz, 
        name="Federal Opportunity Zones (OZ)", 
        style_function=lambda x: {
            'color': '#FFFFFF',
            'weight': 3.5,
            'fillColor': '#FFFFFF',
            'fillOpacity': 0.25
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['TRACT', 'Designation', 'Strategic_Note'],
            aliases=['OZ Tract ID:', 'Boundary Type:', 'Policy Objective:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal;"
        )
    ).add_to(m)

# Layer 7: Economic Anchor Points
if not filtered_infra.empty:
    marker_cluster = MarkerCluster(name=f"{selected_region} Economic Anchors").add_to(m)
    color_map = {'bank': 'darkgreen', 'hospital': 'red', 'childcare': 'orange', 'university': 'purple', 'college': 'purple', 'supermarket': 'blue', 'station': 'gray'}
    icon_map = {'bank': 'bank', 'hospital': 'h-square', 'childcare': 'child', 'university': 'graduation-cap', 'college': 'graduation-cap', 'supermarket': 'shopping-cart', 'station': 'train'}

    for idx, row in filtered_infra.iterrows():
        category = next((row[c] for c in ['amenity', 'shop', 'public_transport'] if c in row and pd.notna(row[c]) and row[c] in infra_options), None)
        if not category: continue
        folium.Marker(
            location=[row.geometry.y, row.geometry.x], 
            popup=row.get('name', category.title()),
            icon=folium.Icon(color=color_map.get(category, 'black'), icon=icon_map.get(category, 'icon'), prefix='fa')
        ).add_to(marker_cluster)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, returned_objects=[], height=700)
