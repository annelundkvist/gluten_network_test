#!/usr/bin/env python3
"""
Gluten Development Analysis App
Analyzes g-force data from mixing machine to understand gluten network formation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
import streamlit as st
from datetime import datetime
import seaborn as sns

# Set up the Streamlit app
st.set_page_config(page_title="Gluten Udvikling Analyse", layout="wide")

def load_data(file_path):
    """Load the CSV data"""
    df = pd.read_csv(file_path)
    return df

def create_spectrogram(data, time, sampling_rate):
    """Create spectrogram from g-force data"""
    # Calculate spectrogram
    frequencies, times, Sxx = spectrogram(data, fs=sampling_rate, 
                                         window='hann', nperseg=1024, 
                                         noverlap=512)
    return frequencies, times, Sxx

def analyze_gluten_development(df):
    """Analyze gluten development patterns"""
    # Calculate statistics
    total_time = df['time'].max()
    mean_gforce = df['gFTotal'].mean()
    std_gforce = df['gFTotal'].std()
    
    # Find peaks in g-force (indicating mixing events)
    peaks, _ = signal.find_peaks(df['gFTotal'], height=mean_gforce + 0.5*std_gforce)
    
    # Analyze frequency content
    sampling_rate = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    
    # Calculate power spectral density
    freqs, psd = signal.welch(df['gFTotal'], fs=sampling_rate, nperseg=1024)
    
    # Detect mixing phases based on g-force variance
    window_size = int(sampling_rate * 10)  # 10 second windows
    rolling_std = df['gFTotal'].rolling(window=window_size, center=True).std()
    
    analysis = {
        'total_time': total_time,
        'mean_gforce': mean_gforce,
        'std_gforce': std_gforce,
        'num_peaks': len(peaks),
        'sampling_rate': sampling_rate,
        'dominant_freq': freqs[np.argmax(psd)],
        'rolling_std': rolling_std
    }
    
    return analysis

def plot_spectrogram(df, analysis):
    """Create spectrogram plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time series plot
    ax1.plot(df['time'], df['gFTotal'], alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Tid (s)')
    ax1.set_ylabel('Total G-kraft')
    ax1.set_title('G-kraft over tid')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram
    frequencies, times, Sxx = create_spectrogram(df['gFTotal'].values, 
                                                df['time'].values, 
                                                analysis['sampling_rate'])
    
    im = ax2.pcolormesh(times, frequencies[:50], 10 * np.log10(Sxx[:50]), 
                       shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frekvens (Hz)')
    ax2.set_xlabel('Tid (s)')
    ax2.set_title('Spektrogram (0-5 Hz)')
    plt.colorbar(im, ax=ax2, label='Power/Frequency (dB/Hz)')
    
    # Power spectral density
    freqs, psd = signal.welch(df['gFTotal'], fs=analysis['sampling_rate'], nperseg=1024)
    ax3.semilogy(freqs[:100], psd[:100])
    ax3.set_xlabel('Frekvens (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Frekvens spektrum')
    ax3.grid(True, alpha=0.3)
    
    # Gluten development indicator (rolling standard deviation)
    ax4.plot(df['time'], analysis['rolling_std'], color='red', linewidth=2)
    ax4.set_xlabel('Tid (s)')
    ax4.set_ylabel('Rullende standardafvigelse')
    ax4.set_title('Gluten udviklings indikator')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🥖 Gluten Udvikling Analyse")
    st.markdown("Analyse af g-kraft data fra røremaskine for glutennetværk udvikling")
    
    # Load data
    try:
        df = load_data('gForce_2025-05-22_12-09-20.csv')
        
        # Perform analysis
        analysis = analyze_gluten_development(df)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total tid", f"{analysis['total_time']:.1f} s")
        
        with col2:
            st.metric("Gennemsnitlig G-kraft", f"{analysis['mean_gforce']:.3f}")
        
        with col3:
            st.metric("Dominerende frekvens", f"{analysis['dominant_freq']:.2f} Hz")
        
        with col4:
            st.metric("Antal peaks", analysis['num_peaks'])
        
        # Create and display plots
        fig = plot_spectrogram(df, analysis)
        st.pyplot(fig)
        
        # Analysis insights
        st.subheader("📊 Analyse Indsigter")
        
        # Mixing phases analysis
        st.markdown("### Rørefaser")
        high_activity_periods = analysis['rolling_std'] > analysis['rolling_std'].quantile(0.75)
        
        if high_activity_periods.any():
            st.success(f"Højaktivitets røreperioder detekteret: {high_activity_periods.sum()} datapunkter")
            st.markdown("Disse perioder indikerer intens røring som er kritisk for glutenudvikling.")
        else:
            st.warning("Ingen højaktivitets røreperioder detekteret")
        
        # Frequency analysis
        st.markdown("### Frekvens Analyse")
        if analysis['dominant_freq'] > 1:
            st.info(f"Dominerende frekvens på {analysis['dominant_freq']:.2f} Hz indikerer hurtig røring")
        else:
            st.info(f"Dominerende frekvens på {analysis['dominant_freq']:.2f} Hz indikerer langsom røring")
        
        # Gluten development assessment
        st.markdown("### Glutennetværk Vurdering")
        std_coefficient = analysis['std_gforce'] / analysis['mean_gforce']
        
        if std_coefficient > 0.1:
            st.success("🟢 Høj variation i g-kraft indikerer aktiv glutenudvikling")
        elif std_coefficient > 0.05:
            st.warning("🟡 Moderat variation - glutenudvikling i gang")
        else:
            st.error("🔴 Lav variation - muligvis utilstrækkelig røring")
        
        # Recipe context
        st.markdown("### 🍞 Opskrift Kontekst")
        st.markdown("""
        **Sesamboller opskrift fra Meyers:**
        - Dejen kræver kraftig røring for at udvikle glutennetværket
        - Optimal røretid afhænger af dejens konsistens og temperatur
        - G-kraft variationer indikerer dejens modstand og elasticitet
        """)
        
        # Recommendations
        st.markdown("### 💡 Anbefalinger")
        total_minutes = analysis['total_time'] / 60
        
        if total_minutes < 8:
            st.warning("Røretid under 8 minutter - overvej længere røring for bedre glutenudvikling")
        elif total_minutes > 15:
            st.warning("Røretid over 15 minutter - risiko for overrøring")
        else:
            st.success(f"Røretid på {total_minutes:.1f} minutter er optimal for glutenudvikling")
        
    except FileNotFoundError:
        st.error("Kunne ikke finde filen 'gForce_2025-05-22_12-09-20.csv'. Sørg for at den er i samme mappe som denne app.")
    except Exception as e:
        st.error(f"Fejl ved indlæsning af data: {str(e)}")

if __name__ == "__main__":
    main()