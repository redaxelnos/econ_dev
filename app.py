import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
import requests
import numpy as np
import branca.colormap as cm
from shapely.geometry import Point
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Prevent OpenStreetMap from hanging the cloud server
ox.settings.timeout = 20

st.set_page_config(page_title="PA Economic Gap Analysis", layout="wide")
st.title("Pennsylvania Economic & Infrastructure Gap Analysis")

# --- SESSION STATE INITIALIZATION ---
if "selected_geoid" not in st.session_state:
    st.session_state.selected_geoid = "42003010300" 
if "tract_modifications" not in st.session_state:
    st.session_state.tract_modifications = {} 

REGIONS = {
    "Statewide View": {"coords": [40.9, -77.6], "zoom": 7, "query": None},
    "Pittsburgh": {"coords": [40.4406, -79.9959], "zoom": 12, "query": "Allegheny County, Pennsylvania, USA"},
    "Philadelphia": {"coords": [39.9526, -75.1652], "zoom": 12, "query": "Philadelphia County, Pennsylvania, USA"},
    "Harrisburg": {"coords": [40.2732, -76.8867], "zoom": 13, "query": "Dauphin County, Pennsylvania, USA"},
    "Allentown": {"coords": [40.6023, -75.4714], "zoom": 13, "query": "Lehigh County, Pennsylvania, USA"},
    "Erie": {"coords": [42.1292, -80.0851], "zoom": 13, "query": "Erie County, Pennsylvania, USA"},
    "Scranton": {"coords": [41.4090, -75.6624], "zoom": 13, "query": "Lackawanna County, Pennsylvania, USA"},
    "Lancaster": {"coords": [40.0379, -76.3055], "zoom": 13, "query": "Lancaster County, Pennsylvania, USA"}
}

selected_region = st.sidebar.selectbox("Select Target Analysis Region", list(REGIONS.keys()))
region_coords = REGIONS[selected_region]["coords"]
region_zoom = REGIONS[selected_region]["zoom"]
region_query = REGIONS[selected_region]["query"]

# 1. Federal Data (Chunked Memory Optimization)
@st.cache_data
def load_census_data():
    base_url = "https://lehd.ces.census.gov/data/lodes/LODES8/pa"
    xwalk = pd.read_csv(f"{base_url}/pa_xwalk.csv.gz", usecols=['tabblk2020', 'trct'], dtype=str)
    
    tract_21, tract_16 = pd.DataFrame(), pd.DataFrame()
    for chunk in pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2021.csv.gz", usecols=['w_geocode', 'C000', 'CE03'], dtype={'w_geocode': str}, chunksize=50000):
        tract_21 = pd.concat([tract_21, chunk.merge(xwalk, left_on='w_geocode', right_on='tabblk2020').groupby('trct')[['C000', 'CE03']].sum().reset_index()]).groupby('trct').sum().reset_index()
        
    for chunk in pd.read_csv(f"{base_url}/wac/pa_wac_S000_JT00_2016.csv.gz", usecols=['w_geocode', 'C000', 'CE03'], dtype={'w_geocode': str}, chunksize=50000):
        tract_16 = pd.concat([tract_16, chunk.merge(xwalk, left_on='w_geocode', right_on='tabblk2020').groupby('trct')[['C000', 'CE03']].sum().reset_index()]).groupby('trct').sum().reset_index()
    
    del xwalk 
    df_jobs = pd.merge(tract_21, tract_16, on='trct', suffixes=('_21', '_16'), how='outer').fillna(0)
    df_jobs['job_growth'] = df_jobs['C000_21'] - df_jobs['C000_16']
    df_jobs['high_wage_growth'] = df_jobs['CE03_21'] - df_jobs['CE03_16']
    
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_42_tract.zip"
    gdf_tracts = gpd.read_file(tiger_url)
    gdf_tracts['geometry'] = gdf_tracts['geometry'].simplify(tolerance=0.005, preserve_topology=True)
    
    gdf_mapped = gdf_tracts.merge(df_jobs, left_on='GEOID', right_on='trct', how='left')
    for col in ['job_growth', 'high_wage_growth', 'C000_21']: gdf_mapped[col] = gdf_mapped[col].fillna(0)
        
    np.random.seed(42)
    gdf_mapped['baseline_home_value'] = 180000 + (gdf_mapped['high_wage_growth'] * 1200) + (gdf_mapped['C000_21'] * 45)
    gdf_mapped['baseline_home_value'] = gdf_mapped['baseline_home_value'].clip(lower=95000, upper=650000)
    return gdf_mapped

# 2. Capital Velocity Data
@st.cache_data
def load_permit_data():
    wprdc_url = "https://data.wprdc.org/api/3/action/datastore_search"
    params = {'resource_id': '20162fb2-7a09-43c1-b0b3-34e87754d9a8', 'q': 'COMMERCIAL', 'limit': 1000}
    try:
        response = requests.get(wprdc_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        df_permits = pd.DataFrame(response.json()['result']['records'])
        if df_permits.empty: return gpd.GeoDataFrame()
        df_permits['cost'] = pd.to_numeric(df_permits.get('estimated_cost', 0), errors='coerce').fillna(0)
        lat_col = 'latitude' if 'latitude' in df_permits.columns else 'y'
        lon_col = 'longitude' if 'longitude' in df_permits.columns else 'x'
        df_permits = df_permits.dropna(subset=[lat_col, lon_col])
        return gpd.GeoDataFrame(df_permits, geometry=gpd.points_from_xy(df_permits[lon_col], df_permits[lat_col]), crs="EPSG:4326")[lambda x: x['cost'] > 0]
    except: return gpd.GeoDataFrame()

# 3. True Polygon OSM Loaders (No more centroids destroying data)
osm_tags = {'amenity': ['bank', 'hospital', 'childcare', 'university', 'college', 'clinic', 'dentist', 'pharmacy'], 'shop': ['supermarket', 'mall', 'wholesale'], 'healthcare': ['hospital', 'clinic', 'center']}

@st.cache_data
def load_osm_regional(query_str):
    if not query_str: return gpd.GeoDataFrame()
    try:
        gdf = ox.features_from_place(query_str, tags=osm_tags)
        if not gdf.empty: return gdf.to_crs(epsg=4326)
    except: pass
    return gpd.GeoDataFrame()

@st.cache_data
def load_osm_tract(wkt_polygon):
    from shapely import wkt
    try:
        gdf = ox.features_from_polygon(wkt.loads(wkt_polygon), tags=osm_tags)
        if not gdf.empty: return gdf.to_crs(epsg=4326)
    except: pass
    return gpd.GeoDataFrame()

# 4. Federal Boundaries
@st.cache_data
def load_federal_boundaries(layer_type):
    url = "https://services6.arcgis.com/zDzo4EZXf1AjkPjO/ArcGIS/rest/services/Qualified_Census_Tracts_2025/FeatureServer/0/query" if layer_type == "QCT" else "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/Opportunity_Zones/FeatureServer/0/query"
    out_fields = "GEOID,TRACT,NAME" if layer_type == "QCT" else "TRACT,STATE_NAME"
    where_clause = "GEOID LIKE '42%'" if layer_type == "QCT" else "TRACT LIKE '42%'"
    all_features = []
    offset = 0
    while offset < 4000:
        params = {"where": where_clause, "outFields": out_fields, "resultRecordCount": 1000, "resultOffset": offset, "f": "geojson"}
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if response.ok:
                features = response.json().get("features", [])
                if not features: break
                all_features.extend(features)
                if len(features) < 1000: break
                offset += 1000
            else: break
        except: break
    if all_features:
        gdf = gpd.GeoDataFrame.from_features({"type": "FeatureCollection", "features": all_features}, crs="EPSG:4326")
        if not gdf.empty:
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.005, preserve_topology=True)
            gdf['Designation'] = 'HUD Distressed Area (QCT)' if layer_type == "QCT" else 'Federal Opportunity Zone'
            return gdf
    return gpd.GeoDataFrame()

# --- INITIALIZE & LOAD DATA ---
with st.spinner(f"Loading {selected_region} Data & Compiling Spatial Matrices..."):
    try:
        gdf_mapped = load_census_data()
        gdf_qct = load_federal_boundaries("QCT")
        gdf_oz = load_federal_boundaries("OZ")
        gdf_permits = load_permit_data() if selected_region == "Pittsburgh" else gpd.GeoDataFrame()
        
        # Smart OSM Loader: Try regional, if empty (Statewide View), load local buffered tract
        gdf_infra = load_osm_regional(region_query)
        if gdf_infra.empty and not gdf_mapped.empty:
            active_row = gdf_mapped[gdf_mapped['GEOID'] == st.session_state.selected_geoid]
            if not active_row.empty:
                # Buffer the active tract by 500 meters to catch hospitals straddling the border
                buffered_wkt = gpd.GeoSeries([active_row.geometry.iloc[0]], crs="EPSG:4326").to_crs(epsg=3857).buffer(500).to_crs(epsg=4326).iloc[0].wkt
                gdf_infra = load_osm_tract(buffered_wkt)
    except Exception as e:
        st.error(f"Failed to load core data: {e}")
        st.stop()

# --- DYNAMIC SPATIAL SPILLOVER HALO CALCULATION ---
halo_radius_m, base_radius_m, spillover_geoids, zone_type, halo_radius_miles = 0, 0, [], "Unknown", 0
primary_geoid = st.session_state.selected_geoid
active_mods = st.session_state.tract_modifications.get(primary_geoid, [])

if active_mods:
    for anchor in active_mods:
        if "Large" in anchor: base_radius_m = max(base_radius_m, 8046) 
        elif "Medium" in anchor: base_radius_m = max(base_radius_m, 3218)
        elif "Small" in anchor: base_radius_m = max(base_radius_m, 1609)
        
    if base_radius_m > 0 and not gdf_mapped.empty:
        try:
            active_tract_geom = gdf_mapped[gdf_mapped['GEOID'] == primary_geoid]
            if not active_tract_geom.empty:
                target_geom_3857 = active_tract_geom.to_crs(epsg=3857).geometry.iloc[0]
                tract_area_sqkm = target_geom_3857.area / 1e6
                if tract_area_sqkm < 2.0: zone_type, multiplier = "Dense Urban", 0.70
                elif tract_area_sqkm < 10.0: zone_type, multiplier = "Suburban", 1.0
                elif tract_area_sqkm < 50.0: zone_type, multiplier = "Exurban / Rural", 2.0
                else: zone_type, multiplier = "Deep Rural", 4.0
                
                halo_radius_m = base_radius_m * multiplier
                halo_radius_miles = round(halo_radius_m / 1609.34, 1)
                buffer_4326 = gpd.GeoSeries([target_geom_3857], crs="EPSG:3857").buffer(halo_radius_m).to_crs(epsg=4326).iloc[0]
                spillover_geoids = gdf_mapped[(gdf_mapped.intersects(buffer_4326)) & (gdf_mapped['GEOID'] != primary_geoid)]['GEOID'].astype(str).tolist()
        except: pass

# --- DIAGNOSTIC EVALUATION ---
qct_ids = set(gdf_qct['GEOID'].astype(str)) if not gdf_qct.empty and 'GEOID' in gdf_qct.columns else set()
oz_ids = set(gdf_oz['TRACT'].astype(str)) if not gdf_oz.empty and 'TRACT' in gdf_oz.columns else set()
high_wage_threshold = np.percentile(gdf_mapped['high_wage_growth'].dropna(), 75)

def evaluate_investment_risk(row):
    geoid, growth, high_wage = str(row['GEOID']), row['job_growth'], row['high_wage_growth']
    is_distressed = (geoid in qct_ids) or (geoid in oz_ids) or any(geoid.endswith(t) for t in oz_ids)
    job_str, hw_str = f"Net job change: {int(growth):+d}", f"High-Wage change: {int(high_wage):+d}"
    if growth <= -30: return f"⚠️ Severely Disadvantaged | {job_str} | {hw_str}."
    elif is_distressed: return f"🔴 High-Risk / Caution | {job_str} | {hw_str}." if growth < -10 else f"🟡 Distressed / Stagnant | {job_str} | {hw_str}." if growth < 20 else f"🟢 Distressed / High-Growth | {job_str} | {hw_str}."
    else: return f"🌟 High Opportunity Hub | {job_str} | {hw_str}." if (high_wage >= high_wage_threshold and growth > 10) else f"⚠️ Declining Standard Tract | {job_str} | {hw_str}." if growth < -10 else f"⚪ Stable | {job_str} | {hw_str}."

gdf_mapped['Investment_Rating'] = gdf_mapped.apply(evaluate_investment_risk, axis=1)

gdf_high_opp = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High Opportunity", na=False)].copy()
gdf_high_risk = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High-Risk", na=False)].copy()

# --- SIDEBAR ---
st.sidebar.markdown("---")
analysis_mode = st.sidebar.radio("Select Map Objective", ["Full Spectrum View (All Tracts)", "⚠️ Severely Disadvantaged Focus", "🚨 Turnaround Target Focus", "🌟 High-Growth Scaling Focus", "🔮 Counterfactual Simulation"])

base_metric = st.sidebar.radio("Base Heatmap Metric", ["Total Job Growth (All Wages)", "High-Wage Job Growth (Exceeding $40k/yr)"])
metric_col = 'job_growth' if "Total" in base_metric else 'high_wage_growth'

if "Severely Disadvantaged" in analysis_mode:
    filtered_tracts = gdf_mapped[gdf_mapped['job_growth'] <= -30]
    default_high_opp, default_high_risk, default_qct, default_oz = False, True, True, True
elif "Turnaround" in analysis_mode:
    filtered_tracts = gdf_mapped[(gdf_mapped[metric_col] < 20) & (gdf_mapped['Investment_Rating'].str.contains("High-Risk|Distressed", na=False))]
    default_high_opp, default_high_risk, default_qct, default_oz = False, True, True, True
elif "High-Growth" in analysis_mode:
    filtered_tracts = gdf_mapped[gdf_mapped['Investment_Rating'].str.contains("High Opportunity|High-Growth", na=False)]
    default_high_opp, default_high_risk, default_qct, default_oz = True, False, False, False
else:
    min_growth = st.sidebar.slider("Minimum Job Growth Threshold", min_value=int(gdf_mapped[metric_col].min()), max_value=int(gdf_mapped[metric_col].max()), value=int(gdf_mapped[metric_col].min()), step=50)
    filtered_tracts = gdf_mapped[gdf_mapped[metric_col] >= min_growth]
    default_high_opp, default_high_risk, default_qct, default_oz = True, True, True, True

show_permits = st.sidebar.checkbox("Overlay Capital Investment Heatmap (WPRDC)", value=True) if selected_region == "Pittsburgh" else False
show_high_opp = st.sidebar.checkbox("High Opportunity Hubs - Neon Yellow", value=default_high_opp)
show_high_risk = st.sidebar.checkbox("High-Risk / Caution Zones - Neon Red", value=default_high_risk)
show_qct = st.sidebar.checkbox("Distressed Areas (HUD QCT) - Neon Blue", value=default_qct)
show_oz = st.sidebar.checkbox("Opportunity Zones (OZ) - Neon White", value=default_oz)

# --- MAP RENDERING ---
try:
    m = folium.Map(location=region_coords, zoom_start=region_zoom, tiles="OpenStreetMap")

    if not filtered_tracts.empty:
        global_p5, global_p95 = np.percentile(gdf_mapped[metric_col].dropna(), [5, 95])
        vmin, vmax = min(global_p5, gdf_mapped[metric_col].min()), max(global_p95, gdf_mapped[metric_col].max())
        if vmin == vmax: vmax += 1
        colormap = cm.LinearColormap(colors=['#d73027', '#fee08b', '#1a9850'], vmin=vmin, vmax=vmax)
        colormap.add_to(m)

        def style_job_base(feature):
            geoid = str(feature['properties'].get('GEOID'))
            is_selected = (geoid == st.session_state.selected_geoid)
            has_mod = geoid in st.session_state.tract_modifications and len(st.session_state.tract_modifications[geoid]) > 0
            is_halo = geoid in spillover_geoids
            
            if has_mod: return {'fillColor': '#00FF00', 'color': '#000000', 'weight': 3.5, 'fillOpacity': 0.85}
            elif is_halo: return {'fillColor': '#00BFFF', 'color': '#00BFFF', 'weight': 2.5, 'dashArray': '5, 5', 'fillOpacity': 0.45}
            elif is_selected: return {'fillColor': '#9400D3', 'color': '#000000', 'weight': 3.5, 'fillOpacity': 0.85}
            else: return {'fillColor': colormap(feature['properties'][metric_col]) if feature['properties'][metric_col] is not None else 'transparent', 'color': '#333333', 'weight': 0.4, 'fillOpacity': 0.75}

        folium.GeoJson(filtered_tracts, name='Job Creation Heatmap', style_function=style_job_base, tooltip=folium.features.GeoJsonTooltip(fields=['GEOID', 'job_growth', 'high_wage_growth', 'baseline_home_value', 'Investment_Rating'], aliases=['Tract:', 'Total Growth:', 'High-Wage Growth:', 'Est. Home Value:', 'Rating:'], style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")).add_to(m)

    sim_group = folium.FeatureGroup(name="Simulated Interventions")
    for geoid, mods in st.session_state.tract_modifications.items():
        if mods:
            tract_geom = gdf_mapped[gdf_mapped['GEOID'].astype(str) == geoid]
            if not tract_geom.empty:
                centroid = tract_geom.geometry.centroid.iloc[0]
                folium.Marker(location=[centroid.y, centroid.x], popup=f"Tract {geoid}: {', '.join(mods)}", icon=folium.Icon(color='green', icon='industry', prefix='fa')).add_to(sim_group)
    sim_group.add_to(m)

    if show_permits and not gdf_permits.empty: HeatMap([[row.geometry.y, row.geometry.x, row['cost']] for idx, row in gdf_permits.iterrows()], radius=15, blur=10).add_to(m)
    if show_high_opp and not gdf_high_opp.empty: folium.GeoJson(gdf_high_opp, style_function=lambda x: {'color': '#FFFF00', 'weight': 3.0, 'fillColor': '#FFFF00', 'fillOpacity': 0.3, 'dashArray': '2, 2'}).add_to(m)
    if show_high_risk and not gdf_high_risk.empty: folium.GeoJson(gdf_high_risk, style_function=lambda x: {'color': '#FF0055', 'weight': 3.5, 'fillColor': '#FF0055', 'fillOpacity': 0.3, 'dashArray': '5, 3'}).add_to(m)
    if show_qct and not gdf_qct.empty: folium.GeoJson(gdf_qct, style_function=lambda x: {'color': '#00FFFF', 'weight': 3.5, 'fillColor': '#00FFFF', 'fillOpacity': 0.25, 'dashArray': '4, 4'}).add_to(m)
    if show_oz and not gdf_oz.empty: folium.GeoJson(gdf_oz, style_function=lambda x: {'color': '#FFFFFF', 'weight': 3.5, 'fillColor': '#FFFFFF', 'fillOpacity': 0.25}).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_output = st_folium(m, use_container_width=True, returned_objects=['last_clicked'], height=500)

    if map_output and map_output.get('last_clicked'):
        click_lat, click_lng = map_output['last_clicked']['lat'], map_output['last_clicked']['lng']
        containing_tract = gdf_mapped[gdf_mapped.contains(Point(click_lng, click_lat))]
        if not containing_tract.empty:
            clicked_geoid = str(containing_tract.iloc[0]['GEOID'])
            if st.session_state.selected_geoid != clicked_geoid:
                st.session_state.selected_geoid = clicked_geoid
                st.rerun()
except Exception as e:
    st.error(f"Map Rendering Error: {str(e)}")

# ==========================================
# --- TRACT INSPECTOR & I-O DASHBOARD ---
# ==========================================
st.markdown("---")
st.markdown(f"### 📍 Interactive Tract Inspector (Selected Tract: `{st.session_state.selected_geoid}`)")

selected_row = gdf_mapped[gdf_mapped['GEOID'].astype(str) == st.session_state.selected_geoid]

if not selected_row.empty:
    row_data = selected_row.iloc[0]
    base_j, base_val = row_data['C000_21'], row_data['baseline_home_value']
    
    # EXACT TRACT-LEVEL INFRASTRUCTURE DETECTION (FAST & ACCURATE)
    detected_features = []
    if not gdf_infra.empty:
        active_geom = selected_row.geometry.iloc[0]
        local_infra = gdf_infra[gdf_infra.intersects(active_geom)]
        for _, row in local_infra.iterrows():
            val = row.get('amenity') or row.get('healthcare') or row.get('shop')
            if pd.notna(val):
                detected_features.append(str(val).replace('_', ' ').title())
    detected_features = list(set(detected_features))
    
    jg = int(row_data['job_growth'])
    if jg <= -50: trend_word, trend_color = "Severe Historical Decline", "red"
    elif jg < 0: trend_word, trend_color = "Contracting Job Market", "orange"
    elif jg == 0: trend_word, trend_color = "Stagnant Market", "gray"
    elif jg < 50: trend_word, trend_color = "Moderate Growth", "green"
    else: trend_word, trend_color = "Rapid Economic Expansion", "green"

    if base_j > 3000: likely_anchor, dream_anchor = "Medium/Regional Hospital", "Mega-Scale Tech Campus & Transit Hub"
    elif base_j > 800: likely_anchor, dream_anchor = "Community Grocery or BRT", "Mega-Scale Hospital"
    else: likely_anchor, dream_anchor = "Childcare or EV Hub", "Regional Advanced Manufacturing"

    col_info, col_controls = st.columns([1, 1.2])
    
    with col_info:
        st.markdown(f"- **Baseline Jobs:** `{int(base_j):,}`\n- **Job Growth (16-21):** `{jg:+d}` (:{trend_color}[{trend_word}])\n- **Est. Home Value:** `${int(base_val):,}`")
        st.markdown(f"**🔍 Existing Infrastructure Context (OSM):**")
        if detected_features:
            for feat in detected_features: st.markdown(f"- ✅ Detected: `{feat}`")
        else:
            st.markdown("- *No major commercial anchors detected via OpenStreetMap in this specific block.*")
        st.success(f"**Highly Probable Fit:** {likely_anchor}")
        st.info(f"**Dream Catalyst Scenario:** {dream_anchor}")

    with col_controls:
        current_mods = st.session_state.tract_modifications.get(st.session_state.selected_geoid, [])
        feature_options = [
            "Small / Community-Scale Hospital / Medical Center", "Medium / Regional-Scale Hospital / Medical Center", "Large / Enterprise Mega-Scale Hospital / Medical Center",
            "Small / Community-Scale Grocery Store / Supermarket", "Medium / Regional-Scale Grocery Store / Supermarket", "Large / Enterprise Mega-Scale Grocery Store / Supermarket",
            "Small / Community-Scale College / University", "Medium / Regional-Scale College / University", "Large / Enterprise Mega-Scale College / University",
            "Small / Community-Scale Fulfillment / Logistics Hub", "Medium / Regional-Scale Fulfillment / Logistics Hub", "Large / Enterprise Mega-Scale Fulfillment / Logistics Hub",
            "Small / Community-Scale Bank / Financial Institution", "Medium / Regional-Scale Bank / Financial Institution", "Large / Enterprise Mega-Scale Bank / Financial Institution",
            "Small / Community-Scale Childcare Facility", "Medium / Regional-Scale Childcare Facility", "Large / Enterprise Mega-Scale Childcare Facility",
            "Small / Community-Scale Advanced Manufacturing", "Medium / Regional-Scale Advanced Manufacturing", "Large / Enterprise Mega-Scale Advanced Manufacturing",
            "Small / Community-Scale Tech / R&D Campus", "Medium / Regional-Scale Tech / R&D Campus", "Large / Enterprise Mega-Scale Tech / R&D Campus",
            "Small / Community EV Charging & Micro-Grid Hub", "Medium / Regional Complete Streets & Pedestrianization", "Medium / Regional Bus Rapid Transit (BRT) Corridor",
            "Large / Enterprise Smart Freight Corridor", "Large / Enterprise High-Speed Rail & Transit Hub"
        ]
        
        with st.form(key=f"sim_form_{st.session_state.selected_geoid}"):
            selected_adds = st.multiselect("Select Anchors to Deploy:", options=feature_options, default=current_mods)
            c1, c2 = st.columns([1, 1])
            run_sim = c1.form_submit_button("🚀 Load Simulation")
            clear_sim = c2.form_submit_button("🗑️ Clear Tract")

        if run_sim:
            st.session_state.tract_modifications[st.session_state.selected_geoid] = selected_adds
            st.rerun()
        if clear_sim:
            st.session_state.tract_modifications[st.session_state.selected_geoid] = []
            st.rerun()

    # --- I-O MATH ---
    st.markdown("---")
    st.markdown("### 📊 Projected Input-Output (I-O) Impact")
    
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
        "Large / Enterprise Mega-Scale Tech / R&D Campus": {"capex": 300, "const": 850, "direct": 950, "indirect": 260, "induced": 750, "tax": 5500000, "retail": 280, "housing": 0.210},
        "Small / Community EV Charging & Micro-Grid Hub": {"capex": 5, "const": 15, "direct": 5, "indirect": 5, "induced": 5, "tax": 40000, "retail": 10, "housing": 0.015},
        "Medium / Regional Complete Streets & Pedestrianization": {"capex": 45, "const": 150, "direct": 20, "indirect": 15, "induced": 50, "tax": 800000, "retail": 350, "housing": 0.060}, 
        "Medium / Regional Bus Rapid Transit (BRT) Corridor": {"capex": 120, "const": 400, "direct": 150, "indirect": 60, "induced": 90, "tax": 1500000, "retail": 200, "housing": 0.080},
        "Large / Enterprise Smart Freight Corridor": {"capex": 600, "const": 1800, "direct": 150, "indirect": 800, "induced": 200, "tax": 5000000, "retail": 100, "housing": 0.050}, 
        "Large / Enterprise High-Speed Rail & Transit Hub": {"capex": 800, "const": 2500, "direct": 400, "indirect": 200, "induced": 300, "tax": 8000000, "retail": 500, "housing": 0.250}
    }
    
    tot_capex, tot_const, tot_direct, tot_indirect, tot_induced, tot_tax, tot_retail, tot_housing_pct = 0, 0, 0, 0, 0, 0, 0, 0.0
    has_commercial, has_transit, max_build_years = False, False, 1 
    job_categories = set()

    if current_mods:
        dream_flags, stretch_flags, plausible_flags = [], [], []
        
        for anchor_name in current_mods:
            if ("Mega-Scale" in anchor_name or "High-Speed Rail" in anchor_name) and base_j < 1500: dream_flags.append(anchor_name)
            elif ("Regional" in anchor_name or "BRT" in anchor_name) and base_j < 500: stretch_flags.append(anchor_name)
            else: plausible_flags.append(anchor_name)

            if "Mega-Scale Hospital" in anchor_name or "High-Speed Rail" in anchor_name or "Mega-Scale College" in anchor_name or "Mega-Scale Advanced Manufacturing" in anchor_name or "Mega-Scale Tech" in anchor_name: max_build_years = max(max_build_years, 5)
            elif "Regional-Scale Hospital" in anchor_name or "Smart Freight Corridor" in anchor_name or "BRT" in anchor_name or "Large / Enterprise" in anchor_name: max_build_years = max(max_build_years, 3)
            elif "Medium / Regional" in anchor_name or "Small / Community-Scale Hospital" in anchor_name or "University" in anchor_name or "Manufacturing" in anchor_name: max_build_years = max(max_build_years, 2)

            if anchor_name in io_matrix:
                if "Transit" in anchor_name or "BRT" in anchor_name or "Complete Streets" in anchor_name or "Freight" in anchor_name: has_transit = True
                elif "Campus" in anchor_name or "Hospital" in anchor_name or "Hub" in anchor_name: has_commercial = True
                
                if "Hospital" in anchor_name: job_categories.update(["Clinical Healthcare", "Medical Admin", "Facilities Ops"])
                elif "Grocery" in anchor_name: job_categories.update(["Retail Sales", "Inventory Mgt", "Customer Service"])
                elif "College" in anchor_name: job_categories.update(["Higher Education", "Research", "Campus Admin"])
                elif "Fulfillment" in anchor_name: job_categories.update(["Logistics", "Warehousing", "Supply Chain Ops"])
                elif "Bank" in anchor_name: job_categories.update(["Financial Services", "Wealth Mgt", "Retail Banking"])
                elif "Childcare" in anchor_name: job_categories.update(["Early Education", "Caregiving"])
                elif "Manufacturing" in anchor_name: job_categories.update(["Precision Manufacturing", "Engineering", "Assembly"])
                elif "Tech" in anchor_name: job_categories.update(["Software Engineering", "R&D", "Data Science"])
                elif "Transit" in anchor_name or "BRT" in anchor_name or "Rail" in anchor_name or "Complete Streets" in anchor_name: job_categories.update(["Transit Operations", "Fleet Maintenance", "Civil Engineering"])

                d = io_matrix[anchor_name]
                tot_capex += d["capex"]; tot_const += d["const"]; tot_direct += d["direct"]
                tot_indirect += d["indirect"]; tot_induced += d["induced"]; tot_tax += d["tax"]
                tot_retail += d["retail"]; tot_housing_pct += d["housing"]

        st.markdown("#### 🎯 Deployment Feasibility Analysis")
        if plausible_flags: st.success(f"**🟢 Plausible Fit:** {', '.join(plausible_flags)}")
        if stretch_flags: st.warning(f"**🟡 Stretch Goal:** {', '.join(stretch_flags)}")
        if dream_flags: st.error(f"**🟣 Dream Scenario:** {', '.join(dream_flags)}")
    
    synergy_active = has_commercial and has_transit
    synergy_multiplier = 1.25 if synergy_active else 1.0

    if synergy_active:
        st.success("🚆 **Transit-Oriented Development (TOD) Synergy Activated!** Labor pool expands by 25%.")

    tot_direct = int(tot_direct * synergy_multiplier)
    tot_indirect = int(tot_indirect * synergy_multiplier)
    tot_induced = int(tot_induced * synergy_multiplier)
    tot_retail = int(tot_retail * synergy_multiplier)
    tot_tax = int(tot_tax * synergy_multiplier)
    tot_housing_pct = tot_housing_pct * synergy_multiplier
            
    local_capture, halo_capture = 0.40, 0.60
    primary_indirect, primary_induced, primary_retail = tot_indirect * local_capture, tot_induced * local_capture, tot_retail * local_capture
    primary_jobs_created = tot_direct + primary_indirect + primary_induced + primary_retail
    primary_proj_jobs = base_j + primary_jobs_created
    primary_proj_val = base_val * (1 + tot_housing_pct)
    
    halo_indirect, halo_induced, halo_retail = tot_indirect * halo_capture, tot_induced * halo_capture, tot_retail * halo_capture
    halo_total_jobs = halo_indirect + halo_induced + halo_retail
    halo_tax = tot_tax * 0.35 
    
    t1, t2, t3 = st.tabs(["📍 Local Host Impact", "🌊 Regional Halo Impact", "⏳ Temporal Impact Horizon"])
    
    with t1:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Est. Capital Investment", f"${tot_capex:,}M")
        d2.metric("Total Local Jobs", f"{int(primary_proj_jobs):,}", delta=f"+{int(primary_jobs_created)} net local lift")
        d3.metric("Host Municipal Tax Lift", f"${tot_tax:,.0f}")
        d4.metric("Est. Host Median Home Value", f"${int(primary_proj_val):,}", delta=f"${int(primary_proj_val - base_val):+,} ({tot_housing_pct*100:+.1f}%)")
        
    with t2:
        if current_mods: st.info(f"The model detected a **{zone_type}** topology, dynamically adjusting the trade area radius to **{halo_radius_miles} miles**.")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Spillover Job Creation", f"+{int(halo_total_jobs)} jobs")
        h2.metric("Halo Retail/Dining Lift", f"+{int(halo_retail)} service jobs")
        h3.metric("Halo Municipal Tax Lift", f"+${halo_tax:,.0f}")
        h4.metric("Secondary Housing Bump", f"+{tot_housing_pct*100*0.35:.1f}% avg lift")
        
    with t3:
        if current_mods:
            core_sectors = ", ".join(list(job_categories)[:6]) if job_categories else "Mixed Commercial Operations"
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"#### Phase 1: Years 0 - {max_build_years} (Development)")
                st.markdown(f"- **Primary Workforce:** Heavy Civil construction, Trades (Steel, Electrical, Carpentry), Architecture & Engineering.")
                st.markdown(f"- **Construction Jobs:** {tot_const:,} (Temporary Peak)")
                st.markdown(f"- **CapEx Deployed:** ${tot_capex}M")
            with c2:
                st.markdown(f"#### Phase 2: Years {max_build_years + 1} - {max_build_years + 3} (Ramp-Up)")
                st.markdown(f"- **Emerging Sectors:** Initial facility hiring focusing on **{core_sectors}**.")
                st.markdown(f"- **Direct Hiring:** {tot_direct:,} jobs")
                st.markdown(f"- **Tax Base:** 50% realization (${tot_tax * 0.5:,.0f})")
            with c3:
                st.markdown(f"#### Phase 3: Years {max_build_years + 4}+ (Maturation)")
                st.markdown(f"- **Dominant Sectors:** Stabilized **{core_sectors}**, supplemented by local services.")
                st.markdown(f"- **Full Stabilization:** {int(primary_jobs_created + halo_total_jobs):,} total jobs.")
                st.markdown(f"- **Full Tax Yield:** ${tot_tax + halo_tax:,.0f} annually.")
