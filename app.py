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
    gdf_permits = load_permit_data() if selected_region == "Pittsburgh" else gpd.GeoDataFrame()

# --- CACHED SPATIAL INFRASTRUCTURE MATCHING ---
@st.cache_data
def get_detected_features(selected_region):
    tract_features = {}
    if selected_region == "Statewide View" or gdf_infra.empty:
        return tract_features
    try:
        infra_proj = gdf_infra.to_crs(gdf_mapped.crs)
        joined = gpd.sjoin(infra_proj, gdf_mapped[['GEOID', 'geometry']], how='inner', predicate='within')
        for geoid, group in joined.groupby('GEOID'):
            amenities = [str(row.get('amenity') or row.get('shop') or row.get('public_transport')) for _, row in group.iterrows() if pd.notna(row.get('amenity') or row.get('shop') or row.get('public_transport'))]
            tract_features[str(geoid)] = list(set(amenities))
    except Exception:
        pass
    return tract_features

tract_detected_features = get_detected_features(selected_region)

# --- SPATIAL SPILLOVER HALO CALCULATION ---
halo_radius_m = 0
spillover_geoids = []
primary_geoid = st.session_state.selected_geoid
active_mods = st.session_state.tract_modifications.get(primary_geoid, [])

if active_mods:
    for anchor in active_mods:
        if "Large" in anchor: halo_radius_m = max(halo_radius_m, 8046) # 5 miles
        elif "Medium" in anchor: halo_radius_m = max(halo_radius_m, 3218) # 2 miles
        elif "Small" in anchor: halo_radius_m = max(halo_radius_m, 1609) # 1 mile
        
    if halo_radius_m > 0 and not gdf_mapped.empty:
        try:
            active_tract_geom = gdf_mapped[gdf_mapped['GEOID'] == primary_geoid]
            if not active_tract_geom.empty:
                target_geom_3857 = active_tract_geom.to_crs(epsg=3857).geometry.iloc[0]
                buffer_geom = target_geom_3857.buffer(halo_radius_m)
                all_3857 = gdf_mapped.to_crs(epsg=3857)
                spillover_mask = all_3857.intersects(buffer_geom) & (all_3857['GEOID'] != primary_geoid)
                spillover_geoids = all_3857[spillover_mask]['GEOID'].astype(str).tolist()
        except Exception:
            pass

# --- PRECISE DIAGNOSTIC EVALUATION ---
qct_ids = set(gdf_qct['GEOID'].astype(str)) if not gdf_qct.empty and 'GEOID' in gdf_qct.columns else set()
oz_ids = set(gdf_oz['TRACT'].astype(str)) if not gdf_oz.empty and 'TRACT' in gdf_oz.columns else set()
high_wage_threshold = np.percentile(gdf_mapped['high_wage_growth'].dropna(), 75)

def evaluate_investment_risk(row):
    geoid, growth, high_wage = str(row['GEOID']), row['job_growth'], row['high_wage_growth']
    is_distressed = (geoid in qct_ids) or (geoid in oz_ids) or any(geoid.endswith(t) for t in oz_ids)
    job_str, hw_str = f"Net job change: {int(growth):+d}", f"High-Wage change: {int(high_wage):+d} (Threshold: +{int(high_wage_threshold)})"
    
    if growth <= -30: return f"⚠️ Severely Disadvantaged / Critical Contraction | {job_str} | {hw_str} (Severe Job Loss)."
    elif is_distressed:
        if growth < -10: return f"🔴 High-Risk / Caution (Distressed + Decline) | {job_str} | {hw_str}."
        elif growth < 20: return f"🟡 Distressed / Stagnant (Needs Catalyst) | {job_str} | {hw_str}."
        else: return f"🟢 Distressed / High-Growth Opportunity | {job_str} | {hw_str}."
    else:
        if high_wage >= high_wage_threshold and growth > 10: return f"🌟 High Opportunity Growth Hub | {job_str} | {hw_str} (Top-Tier Expansion)."
        elif growth < -10: return f"⚠️ Declining Standard Tract | {job_str} | {hw_str} (Severe contraction outside distressed bounds)."
        else: return f"⚪ Stable / Moderate Growth | {job_str} | {hw_str}."

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
        "🌟 High-Growth Scaling Focus (Expansion Hubs Only)",
        "🔮 Counterfactual Impact Simulation (What-If Modeling)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Economic Vitality Layers")
with st.sidebar.expander("ℹ️ Understanding LEHD Data (WAC)", expanded=False):
    st.markdown("""
    - **LEHD (Longitudinal Employer-Household Dynamics):** Tracks jobs where people work, not where they live.
    - **Total Job Growth (All Wages):** Net change in all jobs combined (2016-2021). 
    - **High-Wage Job Growth (Exceeding $40k/yr):** Isolates jobs earning greater than $3,333/month (CE03 tier).
    """)

base_metric = st.sidebar.radio("Base Heatmap Metric (LEHD)", ["Total Job Growth (All Wages)", "High-Wage Job Growth (Exceeding $40k/yr)"])
metric_col = 'job_growth' if base_metric == "Total Job Growth (All Wages)" else 'high_wage_growth'

if analysis_mode == "⚠️ Severely Disadvantaged & High-Need Focus (Critical Intervention)":
    filtered_tracts = gdf_mapped[gdf_mapped['job_growth'] <= -30]
    default_high_opp, default_high_risk, default_qct, default_oz = False, True, True, True
elif analysis_mode == "🚨 Turnaround & Intervention Target Focus (Declining/Distressed Only)":
    filtered_tracts = gdf_mapped[(gdf_mapped[metric_col] < 20) & (gdf_mapped['Investment_Rating'].str.contains("High-Risk|Distressed", na=False))]
    default_high_opp, default_high_risk, default_qct, default_oz = False, True, True, True
elif analysis_mode == "🌟 High-Growth Scaling Focus (Expansion Hubs Only)":
    filtered_tracts = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High Opportunity|High-Growth", na=False)]
    default_high_opp, default_high_risk, default_qct, default_oz = True, False, False, False
else:
    min_growth = st.sidebar.slider("Minimum Job Growth Threshold", min_value=int(gdf_mapped[metric_col].min()), max_value=int(gdf_mapped[metric_col].max()), value=int(gdf_mapped[metric_col].min()), step=50)
    filtered_tracts = gdf_mapped[gdf_mapped[metric_col] >= min_growth]
    default_high_opp, default_high_risk, default_qct, default_oz = True, True, True, True

show_permits = st.sidebar.checkbox("Overlay Capital Investment Heatmap (WPRDC)", value=True) if selected_region == "Pittsburgh" else False

st.sidebar.markdown("---")
st.sidebar.header("Policy & Opportunity Boundaries")
show_high_opp = st.sidebar.checkbox("High Opportunity Hubs - Neon Yellow", value=default_high_opp)
show_high_risk = st.sidebar.checkbox("High-Risk / Caution Zones - Neon Red", value=default_high_risk)
show_qct = st.sidebar.checkbox("Distressed Areas (HUD QCT) - Neon Blue", value=default_qct)
show_oz = st.sidebar.checkbox("Opportunity Zones (OZ) - Neon White", value=default_oz)

# --- RENDER MAP ---
m = folium.Map(location=region_coords, zoom_start=region_zoom, tiles="OpenStreetMap")

if not filtered_tracts.empty:
    global_p5, global_p95 = np.percentile(gdf_mapped[metric_col].dropna(), [5, 95])
    vmin, vmax = min(global_p5, gdf_mapped[metric_col].min()), max(global_p95, gdf_mapped[metric_col].max())
    if vmin == vmax: vmax += 1

    colormap = cm.LinearColormap(colors=['#d73027', '#fee08b', '#1a9850'], vmin=vmin, vmax=vmax)
    colormap.caption = f'Net Growth: {base_metric} (Absolute Statewide Scale)'
    colormap.add_to(m)

    def style_job_base(feature):
        geoid = str(feature['properties'].get('GEOID'))
        is_selected = (geoid == st.session_state.selected_geoid)
        has_mod = geoid in st.session_state.tract_modifications and len(st.session_state.tract_modifications[geoid]) > 0
        is_halo = geoid in spillover_geoids
        
        if has_mod: return {'fillColor': '#00FF00', 'color': '#000000', 'weight': 3.5, 'fillOpacity': 0.85}
        elif is_halo: return {'fillColor': '#00BFFF', 'color': '#00BFFF', 'weight': 2.5, 'dashArray': '5, 5', 'fillOpacity': 0.45}
        elif is_selected: return {'fillColor': '#9400D3', 'color': '#000000', 'weight': 3.5, 'fillOpacity': 0.85}
        else:
            val = feature['properties'][metric_col]
            return {'fillColor': colormap(val) if val is not None else 'transparent', 'color': '#333333', 'weight': 0.4, 'fillOpacity': 0.75}

    folium.GeoJson(
        filtered_tracts,
        style_function=style_job_base,
        tooltip=folium.features.GeoJsonTooltip(fields=['GEOID', 'job_growth', 'high_wage_growth', 'baseline_home_value', 'Investment_Rating'], aliases=['Census Tract ID:', 'Total Job Growth:', 'High-Wage Growth:', 'Est. Median Home Value:', 'Diagnostic Evaluation:'], localize=True, sticky=True, style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; max-width: 280px; word-wrap: break-word; white-space: normal;")
    ).add_to(m)

for geoid, mods in st.session_state.tract_modifications.items():
    if mods:
        tract_geom = gdf_mapped[gdf_mapped['GEOID'].astype(str) == geoid]
        if not tract_geom.empty:
            centroid = tract_geom.geometry.centroid.iloc[0]
            folium.Marker(location=[centroid.y, centroid.x], popup=f"Tract {geoid}: Modifications -> {', '.join(mods)}", icon=folium.Icon(color='green', icon='industry', prefix='fa')).add_to(m)

if show_permits and not gdf_permits.empty:
    HeatMap([[row.geometry.y, row.geometry.x, row['cost']] for idx, row in gdf_permits.iterrows()], radius=15, blur=10, max_zoom=1).add_to(m)
if show_high_opp and not gdf_high_opp.empty: folium.GeoJson(gdf_high_opp, style_function=lambda x: {'color': '#FFFF00', 'weight': 3.0, 'fillColor': '#FFFF00', 'fillOpacity': 0.3, 'dashArray': '2, 2'}).add_to(m)
if show_high_risk and not gdf_high_risk.empty: folium.GeoJson(gdf_high_risk, style_function=lambda x: {'color': '#FF0055', 'weight': 3.5, 'fillColor': '#FF0055', 'fillOpacity': 0.3, 'dashArray': '5, 3'}).add_to(m)
if show_qct and not gdf_qct.empty: folium.GeoJson(gdf_qct, style_function=lambda x: {'color': '#00FFFF', 'weight': 3.5, 'fillColor': '#00FFFF', 'fillOpacity': 0.25, 'dashArray': '4, 4'}).add_to(m)
if show_oz and not gdf_oz.empty: folium.GeoJson(gdf_oz, style_function=lambda x: {'color': '#FFFFFF', 'weight': 3.5, 'fillColor': '#FFFFFF', 'fillOpacity': 0.25}).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

map_output = st_folium(m, use_container_width=True, returned_objects=['last_clicked'], height=600)

if map_output and map_output.get('last_clicked'):
    click_lat, click_lng = map_output['last_clicked']['lat'], map_output['last_clicked']['lng']
    containing_tract = gdf_mapped[gdf_mapped.contains(Point(click_lng, click_lat))]
    if not containing_tract.empty:
        clicked_geoid = str(containing_tract.iloc[0]['GEOID'])
        if st.session_state.selected_geoid != clicked_geoid:
            st.session_state.selected_geoid = clicked_geoid
            st.rerun()

# ==========================================
# --- INTERACTIVE TRACT INSPECTOR ---
# ==========================================
st.markdown("---")
st.markdown(f"### 📍 Interactive Tract Inspector & Feature Toggle Panel (Selected Tract: `{st.session_state.selected_geoid}`)")

selected_row = gdf_mapped[gdf_mapped['GEOID'].astype(str) == st.session_state.selected_geoid]

if not selected_row.empty:
    row_data = selected_row.iloc[0]
    base_j, base_val = row_data['C000_21'], row_data['baseline_home_value']
    detected_features = tract_detected_features.get(st.session_state.selected_geoid, [])
    
    # --- SUITABILITY ENGINE ---
    if base_j > 3000:
        likely_anchor = "Medium / Regional-Scale Hospital or College"
        dream_anchor = "Large / Enterprise Mega-Scale Tech Campus"
    elif base_j > 800:
        likely_anchor = "Small / Community-Scale Grocery Store or Fulfillment Hub"
        dream_anchor = "Large / Enterprise Mega-Scale Hospital"
    else:
        likely_anchor = "Small / Community-Scale Childcare Facility"
        dream_anchor = "Medium / Regional-Scale Advanced Manufacturing"

    col_info, col_controls = st.columns([1, 1.2])
    
    with col_info:
        st.markdown("#### 📋 Baseline Tract Diagnostics")
        st.write(f"- **Census GEOID:** `{st.session_state.selected_geoid}`")
        st.write(f"- **Baseline Workplace Jobs:** `{int(base_j):,}`")
        st.write(f"- **Est. Baseline Home Value:** `${int(base_val):,}`")
        
        st.markdown("#### 🤖 AI Site Suitability Assessment")
        st.success(f"**Highly Probable Fit:** {likely_anchor}")
        st.info(f"**Dream Catalyst Scenario:** {dream_anchor}")

    with col_controls:
        st.markdown("#### 🛠️ Add / Subtract Features & Infrastructure")
        current_mods = st.session_state.tract_modifications.get(st.session_state.selected_geoid, [])
        
        feature_options = [
            "Small / Community-Scale Hospital / Medical Center", "Medium / Regional-Scale Hospital / Medical Center", "Large / Enterprise Mega-Scale Hospital / Medical Center",
            "Small / Community-Scale Grocery Store / Supermarket", "Medium / Regional-Scale Grocery Store / Supermarket", "Large / Enterprise Mega-Scale Grocery Store / Supermarket",
            "Small / Community-Scale College / University", "Medium / Regional-Scale College / University", "Large / Enterprise Mega-Scale College / University",
            "Small / Community-Scale Fulfillment / Logistics Hub", "Medium / Regional-Scale Fulfillment / Logistics Hub", "Large / Enterprise Mega-Scale Fulfillment / Logistics Hub",
            "Small / Community-Scale Bank / Financial Institution", "Medium / Regional-Scale Bank / Financial Institution", "Large / Enterprise Mega-Scale Bank / Financial Institution",
            "Small / Community-Scale Childcare Facility", "Medium / Regional-Scale Childcare Facility", "Large / Enterprise Mega-Scale Childcare Facility",
            "Small / Community-Scale Advanced Manufacturing", "Medium / Regional-Scale Advanced Manufacturing", "Large / Enterprise Mega-Scale Advanced Manufacturing",
            "Small / Community-Scale Tech / R&D Campus", "Medium / Regional-Scale Tech / R&D Campus", "Large / Enterprise Mega-Scale Tech / R&D Campus"
        ]
        
        with st.form(key=f"sim_form_{st.session_state.selected_geoid}"):
            selected_adds = st.multiselect("Select Anchors to Deploy / Simulate:", options=feature_options, default=current_mods)
            submit_col, clear_col = st.columns([1, 1])
            run_sim = submit_col.form_submit_button("🚀 Load Simulation")
            clear_sim = clear_col.form_submit_button("🗑️ Clear Tract")

        if run_sim:
            st.session_state.tract_modifications[st.session_state.selected_geoid] = selected_adds
            st.rerun()
        if clear_sim:
            st.session_state.tract_modifications[st.session_state.selected_geoid] = []
            st.rerun()

    # ==========================================
    # --- DYNAMIC I-O IMPACT & TIME DILATION ---
    # ==========================================
    if current_mods:
        st.markdown("---")
        st.markdown("### 📊 Input-Output (I-O) Regional Economic Impact")
        
        io_matrix = {
            "Small / Community-Scale Hospital / Medical Center": {"capex": 15, "const": 45, "direct": 50, "indirect": 15, "induced": 20, "tax": 450000, "retail": 12, "housing": 0.025},
            "Medium / Regional-Scale Hospital / Medical Center": {"capex": 150, "const": 400, "direct": 300, "indirect": 120, "induced": 150, "tax": 3500000, "retail": 75, "housing": 0.080},
            "Large / Enterprise Mega-Scale Hospital / Medical Center": {"capex": 500, "const": 1400, "direct": 700, "indirect": 350, "induced": 420, "tax": 12000000, "retail": 190, "housing": 0.160},
            "Small / Community-Scale Grocery Store / Supermarket": {"capex": 2, "const": 10, "direct": 25, "indirect": 6, "induced": 10, "tax": 60000, "retail": 5, "housing": 0.015},
            "Medium / Regional-Scale Grocery Store / Supermarket": {"capex": 18, "const": 50, "direct": 90, "indirect": 28, "induced": 35, "tax": 280000, "retail": 22, "housing": 0.045},
            "Large / Enterprise Mega-Scale Grocery Store / Supermarket": {"capex": 45, "const": 120, "direct": 210, "indirect": 75, "induced": 85, "tax": 750000, "retail": 55, "housing": 0.075},
            "Small / Community-Scale College / University": {"capex": 40, "const": 110, "direct": 120, "indirect": 45, "induced": 70, "tax": 150000, "retail": 40, "housing": 0.050},
            "Medium / Regional-Scale College / University": {"capex": 180, "const": 500, "direct": 350, "indirect": 160, "induced": 240, "tax": 600000, "retail": 120, "housing": 0.100},
            "Large / Enterprise Mega-Scale College / University": {"capex": 600, "const": 1800, "direct": 800, "indirect": 420, "induced": 650, "tax": 2200000, "retail": 320, "housing": 0.180},
            "Small / Community-Scale Fulfillment / Logistics Hub": {"capex": 25, "const": 70, "direct": 75, "indirect": 30, "induced": 25, "tax": 400000, "retail": 14, "housing": 0.020},
            "Medium / Regional-Scale Fulfillment / Logistics Hub": {"capex": 120, "const": 350, "direct": 250, "indirect": 115, "induced": 90, "tax": 1800000, "retail": 50, "housing": 0.060},
            "Large / Enterprise Mega-Scale Fulfillment / Logistics Hub": {"capex": 350, "const": 1000, "direct": 500, "indirect": 260, "induced": 190, "tax": 4500000, "retail": 110, "housing": 0.110},
            "Small / Community-Scale Bank / Financial Institution": {"capex": 2, "const": 8, "direct": 6, "indirect": 2, "induced": 3, "tax": 25000, "retail": 1, "housing": 0.005},
            "Medium / Regional-Scale Bank / Financial Institution": {"capex": 10, "const": 30, "direct": 22, "indirect": 8, "induced": 10, "tax": 150000, "retail": 4, "housing": 0.020},
            "Large / Enterprise Mega-Scale Bank / Financial Institution": {"capex": 75, "const": 250, "direct": 180, "indirect": 75, "induced": 90, "tax": 1200000, "retail": 25, "housing": 0.055},
            "Small / Community-Scale Childcare Facility": {"capex": 0.3, "const": 3, "direct": 8, "indirect": 2, "induced": 3, "tax": 8000, "retail": 2, "housing": 0.010},
            "Medium / Regional-Scale Childcare Facility": {"capex": 2, "const": 15, "direct": 28, "indirect": 8, "induced": 12, "tax": 45000, "retail": 6, "housing": 0.025},
            "Large / Enterprise Mega-Scale Childcare Facility": {"capex": 8, "const": 40, "direct": 85, "indirect": 25, "induced": 35, "tax": 180000, "retail": 18, "housing": 0.050},
            "Small / Community-Scale Advanced Manufacturing": {"capex": 20, "const": 60, "direct": 40, "indirect": 35, "induced": 20, "tax": 200000, "retail": 8, "housing": 0.030},
            "Medium / Regional-Scale Advanced Manufacturing": {"capex": 85, "const": 250, "direct": 180, "indirect": 160, "induced": 85, "tax": 900000, "retail": 35, "housing": 0.075},
            "Large / Enterprise Mega-Scale Advanced Manufacturing": {"capex": 400, "const": 1100, "direct": 600, "indirect": 550, "induced": 320, "tax": 3800000, "retail": 140, "housing": 0.140},
            "Small / Community-Scale Tech / R&D Campus": {"capex": 12, "const": 35, "direct": 60, "indirect": 15, "induced": 45, "tax": 350000, "retail": 18, "housing": 0.040},
            "Medium / Regional-Scale Tech / R&D Campus": {"capex": 80, "const": 220, "direct": 300, "indirect": 80, "induced": 240, "tax": 1600000, "retail": 90, "housing": 0.110},
            "Large / Enterprise Mega-Scale Tech / R&D Campus": {"capex": 300, "const": 850, "direct": 950, "indirect": 260, "induced": 750, "tax": 5500000, "retail": 280, "housing": 0.210}
        }
        
        tot_capex, tot_const, tot_direct, tot_indirect, tot_induced, tot_tax, tot_retail, tot_housing_pct = 0, 0, 0, 0, 0, 0, 0, 0.0
        
        for anchor_name in current_mods:
            if anchor_name in io_matrix:
                d = io_matrix[anchor_name]
                
                # Feasibility Grading Overlay
                if "Mega-Scale" in anchor_name and base_j < 1000: grade = "🟣 Dream Scenario"
                elif "Regional" in anchor_name and base_j < 500: grade = "🟡 Stretch Goal"
                else: grade = "🟢 Plausible/Likely"
                st.caption(f"**{anchor_name}** — AI Feasibility: {grade}")
                
                tot_capex += d["capex"]; tot_const += d["const"]; tot_direct += d["direct"]
                tot_indirect += d["indirect"]; tot_induced += d["induced"]; tot_tax += d["tax"]
                tot_retail += d["retail"]; tot_housing_pct += d["housing"]
                
        local_capture, halo_capture = 0.40, 0.60
        primary_indirect, primary_induced, primary_retail = tot_indirect * local_capture, tot_induced * local_capture, tot_retail * local_capture
        primary_jobs_created = tot_direct + primary_indirect + primary_induced + primary_retail
        primary_proj_jobs = base_j + primary_jobs_created
        primary_proj_val = base_val * (1 + tot_housing_pct)
        
        halo_indirect, halo_induced, halo_retail = tot_indirect * halo_capture, tot_induced * halo_capture, tot_retail * halo_capture
        halo_total_jobs = halo_indirect + halo_induced + halo_retail
        halo_tax = tot_tax * 0.35 
        
        t1, t2, t3 = st.tabs(["📍 Local Host Impact", "🌊 Regional Halo Impact", "⏳ Temporal Impact Horizon (Time Dilation)"])
        
        with t1:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Est. Capital Investment (CapEx)", f"${tot_capex:,}M")
            d2.metric("Total Local Jobs (I-O)", f"{int(primary_proj_jobs):,}", delta=f"+{int(primary_jobs_created)} net local lift")
            d3.metric("Host Municipal Tax Lift", f"${tot_tax:,.0f}")
            d4.metric("Est. Host Median Home Value", f"${int(primary_proj_val):,}", delta=f"${int(primary_proj_val - base_val):+,} ({tot_housing_pct*100:+.1f}%)")
            
        with t2:
            st.info(f"Economic effects radiate to **{len(spillover_geoids)}** neighboring tracts in a {round(halo_radius_m/1609.34, 1)}-mile radius (dashed blue on map).")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Spillover Job Creation", f"+{int(halo_total_jobs)} jobs")
            h2.metric("Halo Retail/Dining Lift", f"+{int(halo_retail)} service jobs")
            h3.metric("Halo Municipal Tax Lift", f"+${halo_tax:,.0f}")
            h4.metric("Secondary Housing Bump", f"+{tot_housing_pct*100*0.35:.1f}% avg lift")
            
        with t3:
            st.markdown("Economic impacts do not materialize instantly. Below is the probabilistic realization timeframe:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### Phase 1: Years 0 - 2")
                st.markdown(f"- **Construction Jobs:** {tot_const:,} (Peak)")
                st.markdown(f"- **CapEx Deployed:** ${tot_capex}M")
                st.markdown("- **Operational Jobs:** 0")
                st.markdown("- **Housing Impact:** Speculative bump (+1%)")
            with c2:
                st.markdown("#### Phase 2: Years 3 - 5")
                st.markdown(f"- **Direct Hiring:** {tot_direct:,} jobs (Ramp-up)")
                st.markdown(f"- **Tax Base:** 50% realization (${tot_tax * 0.5:,.0f})")
                st.markdown(f"- **Halo Spillover:** Supply chains form.")
                st.markdown(f"- **Housing Impact:** Accelerated growth.")
            with c3:
                st.markdown("#### Phase 3: Years 5 - 10+")
                st.markdown(f"- **Full Stabilization:** {int(primary_jobs_created + halo_total_jobs):,} total regional jobs.")
                st.markdown(f"- **Full Tax Yield:** ${tot_tax + halo_tax:,.0f} annually.")
                st.markdown(f"- **Agglomeration:** Surrounding retail & services fully matured.")
