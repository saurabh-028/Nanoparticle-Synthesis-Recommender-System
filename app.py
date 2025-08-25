import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="Nanoparticle Parameter Recommendation System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_excel('data.xlsx')

# Initialize scalers
@st.cache_resource
def initialize_scalers():
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[['DLS_norm', 'UVVIS_norm']] = scaler.fit_transform(df[['DLS (nm)', 'UV VIS']])
    return scaler, df_scaled

output_scaler, df_scaled = initialize_scalers()

def recommend_by_combined_score(dls_target, uvvis_target, top_n=5, alpha=0.5):
    """
    Recommend parameters based on combined cosine similarity and euclidean distance
    """
    target_df = pd.DataFrame([[dls_target, uvvis_target]], columns=['DLS (nm)', 'UV VIS'])
    target_norm = output_scaler.transform(target_df)
    target_vector = target_norm.reshape(1, -1)

    sim = cosine_similarity(target_vector, df_scaled[['DLS_norm', 'UVVIS_norm']].values)[0]
    dist = euclidean_distances(target_vector, df_scaled[['DLS_norm', 'UVVIS_norm']].values)[0]

    # Normalize both to 0–1 range
    sim_norm = (sim - np.min(sim)) / (np.max(sim) - np.min(sim)) if np.max(sim) != np.min(sim) else sim
    dist_norm = (dist - np.min(dist)) / (np.max(dist) - np.min(dist)) if np.max(dist) != np.min(dist) else dist

    # Combined score: higher is better
    df_temp = df.copy()
    df_temp['score'] = alpha * sim_norm + (1 - alpha) * (1 - dist_norm)
    top_recommendations = df_temp.sort_values(by='score', ascending=False).head(top_n)

    return top_recommendations[[
        'Time (min)', 'Scanspeed (mm/s)', 'Fluence (J/cm2)',
        'DLS (nm)', 'UV VIS', 'score'
    ]]

def inverse_knn_predictor(dls_target, uvvis_target, k=9):
    """
    Predicts [Time, Scanspeed, Fluence] using KNN given desired [DLS, UV VIS].
    """
    X = df[['DLS (nm)', 'UV VIS']]
    y = df[['Time (min)', 'Scanspeed (mm/s)', 'Fluence (J/cm2)']]

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)

    target = np.array([[dls_target, uvvis_target]])
    predicted = model.predict(target)

    return pd.DataFrame(predicted, columns=['Time (min)', 'Scanspeed (mm/s)', 'Fluence (J/cm2)'])

# Main App
def main():
    st.markdown('<h1 class="main-header">🔬 Nanoparticle Parameter Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>
    Get optimized processing parameters for your desired DLS size and UV-VIS absorption
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for inputs
    with st.sidebar:
        st.header("🎯 Target Properties")
        
        # Input parameters
        dls_target = st.number_input(
            "Target DLS Size (nm) [52.7 - 239.8]",
            min_value=float(df['DLS (nm)'].min()),
            max_value=float(df['DLS (nm)'].max()),
            value=100.0,
            step=1.0,
            help="Desired dynamic light scattering particle size"
        )
        
        uvvis_target = st.number_input(
            "Target UV-VIS Absorption [0.219 - 2.541]",
            min_value=float(df['UV VIS'].min()),
            max_value=float(df['UV VIS'].max()),
            value=0.5,
            step=0.05,
            help="Desired UV-VIS absorption value"
        )
        
        st.subheader("⚙️ Algorithm Settings")
        
        # Algorithm parameters
        alpha = st.slider(
            "Similarity Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Balance between cosine similarity and distance (0=distance only, 1=similarity only)"
        )
        
        top_n = st.selectbox(
            "Number of Recommendations",
            options=[3, 5, 7, 10],
            index=1,
            help="Number of top recommendations to display"
        )
        
        k_neighbors = st.selectbox(
            "KNN Neighbors",
            options=[3, 5, 7, 9, 11],
            index=2,
            help="Number of neighbors for KNN prediction"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Data Overview")
        
        # Data statistics
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📏 DLS Size Range</h4>
                <p>{df['DLS (nm)'].min():.1f} - {df['DLS (nm)'].max():.1f} nm</p>
            </div>
            """, unsafe_allow_html=True)
            
        with stats_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔍 UV-VIS Range</h4>
                <p>{df['UV VIS'].min():.2f} - {df['UV VIS'].max():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset size
        st.info(f"Dataset contains {len(df)} experimental data points")
        
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            df, 
            x='Time (min)', 
            y='Scanspeed (mm/s)', 
            z='Fluence (J/cm2)',
            color='DLS (nm)',
            size='UV VIS',
            title="3D Parameter Space Visualization",
            color_continuous_scale="viridis"
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("🎯 Target Visualization")
        
        # Target point visualization
        fig_target = go.Figure()
        
        # Add all data points
        fig_target.add_trace(go.Scatter(
            x=df['DLS (nm)'],
            y=df['UV VIS'],
            mode='markers',
            name='Dataset',
            marker=dict(
                color='lightblue',
                size=8,
                opacity=0.6
            )
        ))
        
        # Add target point
        fig_target.add_trace(go.Scatter(
            x=[dls_target],
            y=[uvvis_target],
            mode='markers',
            name='Target',
            marker=dict(
                color='red',
                size=15,
                symbol='star'
            )
        ))
        
        fig_target.update_layout(
            title="Target vs Dataset Distribution",
            xaxis_title="DLS Size (nm)",
            yaxis_title="UV-VIS Absorption",
            height=400
        )
        
        st.plotly_chart(fig_target, use_container_width=True)

    # Generate recommendations
    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Analyzing data and generating recommendations..."):
            # Get recommendations
            recommendations = recommend_by_combined_score(
                dls_target, uvvis_target, top_n, alpha
            )
            
            # Get KNN prediction
            knn_prediction = inverse_knn_predictor(
                dls_target, uvvis_target, k_neighbors
            )
            
            st.success("✅ Recommendations generated successfully!")
            
            # Display results
            st.header("📋 Recommendations")
            
            # KNN Prediction (top recommendation)
            st.subheader("🎯 KNN Predicted Parameters")
            
            knn_col1, knn_col2, knn_col3 = st.columns(3)
            
            with knn_col1:
                st.metric(
                    "⏱️ Time (min)",
                    f"{knn_prediction.iloc[0]['Time (min)']:.2f}",
                    help="Recommended processing time"
                )
                
            with knn_col2:
                st.metric(
                    "⚡ Scan Speed (mm/s)",
                    f"{knn_prediction.iloc[0]['Scanspeed (mm/s)']:.2f}",
                    help="Recommended scan speed"
                )
                
            with knn_col3:
                st.metric(
                    "💥 Fluence (J/cm²)",
                    f"{knn_prediction.iloc[0]['Fluence (J/cm2)']:.2f}",
                    help="Recommended laser fluence"
                )
            
            # Top recommendations table
            st.subheader("📊 Top Similar Experiments")
            
            # Format the recommendations dataframe for better display
            display_df = recommendations.copy()
            for col in ['Time (min)', 'Scanspeed (mm/s)', 'Fluence (J/cm2)', 'DLS (nm)', 'UV VIS']:
                display_df[col] = display_df[col].round(2)
            display_df['score'] = display_df['score'].round(4)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization of recommendations
            st.subheader("📈 Recommendation Analysis")
            
            # Create subplots
            fig_rec = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Time vs Score', 'Scan Speed vs Score', 
                              'Fluence vs Score', 'Parameter Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "bar"}]]
            )
            
            # Time vs Score
            fig_rec.add_trace(
                go.Scatter(x=recommendations['Time (min)'], 
                          y=recommendations['score'],
                          mode='markers+lines',
                          name='Time'),
                row=1, col=1
            )
            
            # Scan Speed vs Score
            fig_rec.add_trace(
                go.Scatter(x=recommendations['Scanspeed (mm/s)'], 
                          y=recommendations['score'],
                          mode='markers+lines',
                          name='Scan Speed'),
                row=1, col=2
            )
            
            # Fluence vs Score
            fig_rec.add_trace(
                go.Scatter(x=recommendations['Fluence (J/cm2)'], 
                          y=recommendations['score'],
                          mode='markers+lines',
                          name='Fluence'),
                row=2, col=1
            )
            
            # Parameter comparison bar chart
            params = ['Time (min)', 'Scanspeed (mm/s)', 'Fluence (J/cm2)']
            top_rec = recommendations.iloc[0]
            values = [top_rec[param] for param in params]
            
            fig_rec.add_trace(
                go.Bar(x=params, y=values, name='Top Recommendation'),
                row=2, col=2
            )
            
            fig_rec.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_rec, use_container_width=True)
            
            # Download recommendations
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=csv,
                file_name=f"recommendations_DLS{dls_target}_UVVIS{uvvis_target}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()