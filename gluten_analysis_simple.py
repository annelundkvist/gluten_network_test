#!/usr/bin/env python3
"""
Simple Gluten Development Analysis
Creates spectrogram and analysis without Streamlit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
import seaborn as sns

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path):
    """Load the CSV data"""
    df = pd.read_csv(file_path)
    return df

def create_spectrogram(data, sampling_rate):
    """Create spectrogram from g-force data"""
    frequencies, times, Sxx = spectrogram(data, fs=sampling_rate, 
                                         window='hann', nperseg=1024, 
                                         noverlap=512)
    return frequencies, times, Sxx

def analyze_gluten_development(df):
    """Analyze gluten development patterns"""
    total_time = df['time'].max()
    mean_gforce = df['gFTotal'].mean()
    std_gforce = df['gFTotal'].std()
    
    # Find peaks in g-force
    peaks, _ = signal.find_peaks(df['gFTotal'], height=mean_gforce + 0.5*std_gforce)
    
    # Calculate sampling rate
    sampling_rate = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    
    # Power spectral density
    freqs, psd = signal.welch(df['gFTotal'], fs=sampling_rate, nperseg=1024)
    
    # Rolling standard deviation (gluten development indicator)
    window_size = int(sampling_rate * 10)  # 10 second windows
    rolling_std = df['gFTotal'].rolling(window=window_size, center=True).std()
    
    analysis = {
        'total_time': total_time,
        'mean_gforce': mean_gforce,
        'std_gforce': std_gforce,
        'num_peaks': len(peaks),
        'sampling_rate': sampling_rate,
        'dominant_freq': freqs[np.argmax(psd)],
        'rolling_std': rolling_std,
        'freqs': freqs,
        'psd': psd
    }
    
    return analysis

def create_analysis_plots(df, analysis):
    """Create comprehensive analysis plots"""
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle('Gluten Udvikling Analyse - Sesamboller Dej', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series
    ax1 = plt.subplot(3, 3, (1, 2))
    ax1.plot(df['time']/60, df['gFTotal'], alpha=0.8, linewidth=0.8, color='blue')
    ax1.set_xlabel('Tid (minutter)')
    ax1.set_ylabel('Total G-kraft')
    ax1.set_title('G-kraft over tid')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spectrogram
    ax2 = plt.subplot(3, 3, (4, 5))
    frequencies, times, Sxx = create_spectrogram(df['gFTotal'].values, analysis['sampling_rate'])
    im = ax2.pcolormesh(times/60, frequencies[:50], 10 * np.log10(Sxx[:50]), 
                       shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frekvens (Hz)')
    ax2.set_xlabel('Tid (minutter)')
    ax2.set_title('Spektrogram (0-5 Hz)')
    plt.colorbar(im, ax=ax2, label='Power/Frequency (dB/Hz)')
    
    # Plot 3: Power spectral density
    ax3 = plt.subplot(3, 3, 3)
    ax3.semilogy(analysis['freqs'][:100], analysis['psd'][:100], color='red')
    ax3.set_xlabel('Frekvens (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Frekvens spektrum')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gluten development indicator
    ax4 = plt.subplot(3, 3, 6)
    ax4.plot(df['time']/60, analysis['rolling_std'], color='green', linewidth=2)
    ax4.set_xlabel('Tid (minutter)')
    ax4.set_ylabel('Rullende std')
    ax4.set_title('Gluten udviklings indikator')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 3D accelerometer data
    ax5 = plt.subplot(3, 3, (7, 8))
    ax5.plot(df['time']/60, df['gFx'], alpha=0.7, label='X-akse', linewidth=0.8)
    ax5.plot(df['time']/60, df['gFy'], alpha=0.7, label='Y-akse', linewidth=0.8)
    ax5.plot(df['time']/60, df['gFz'], alpha=0.7, label='Z-akse', linewidth=0.8)
    ax5.set_xlabel('Tid (minutter)')
    ax5.set_ylabel('G-kraft')
    ax5.set_title('3D accelerometer data')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Statistics summary
    ax6 = plt.subplot(3, 3, 9)
    ax6.axis('off')
    
    # Calculate additional metrics
    total_minutes = analysis['total_time'] / 60
    std_coefficient = analysis['std_gforce'] / analysis['mean_gforce']
    high_activity_periods = (analysis['rolling_std'] > analysis['rolling_std'].quantile(0.75)).sum()
    
    # Create text summary
    summary_text = f"""
ANALYSE RESULTATER:

Total røretid: {total_minutes:.1f} min
Gennemsnitlig G-kraft: {analysis['mean_gforce']:.3f}
Standard afvigelse: {analysis['std_gforce']:.3f}
Variations koefficient: {std_coefficient:.3f}

Dominerende frekvens: {analysis['dominant_freq']:.2f} Hz
Antal peaks: {analysis['num_peaks']}
Højaktivitets perioder: {high_activity_periods}

VURDERING:
"""
    
    # Add assessment
    if std_coefficient > 0.1:
        summary_text += "✓ Høj variation - god glutenudvikling\n"
    elif std_coefficient > 0.05:
        summary_text += "△ Moderat variation - delvis glutenudvikling\n"
    else:
        summary_text += "✗ Lav variation - utilstrækkelig røring\n"
    
    if 8 <= total_minutes <= 15:
        summary_text += "✓ Optimal røretid for glutennetværk\n"
    elif total_minutes < 8:
        summary_text += "△ Kort røretid - overvej længere røring\n"
    else:
        summary_text += "△ Lang røretid - risiko for overrøring\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the analysis"""
    try:
        print("🥖 Indlæser gluten udvikling data...")
        df = load_data('gForce_2025-05-22_12-09-20.csv')
        
        print("📊 Udfører analyse...")
        analysis = analyze_gluten_development(df)
        
        print("📈 Opretter plots...")
        fig = create_analysis_plots(df, analysis)
        
        # Save the plot
        plt.savefig('gluten_analysis_results.png', dpi=300, bbox_inches='tight')
        print("💾 Resultater gemt som 'gluten_analysis_results.png'")
        
        # Show the plot
        plt.show()
        
        # Print summary to console
        print("\n" + "="*50)
        print("GLUTEN UDVIKLING ANALYSE SAMMENFATNING")
        print("="*50)
        print(f"Total røretid: {analysis['total_time']/60:.1f} minutter")
        print(f"Gennemsnitlig G-kraft: {analysis['mean_gforce']:.3f}")
        print(f"Dominerende frekvens: {analysis['dominant_freq']:.2f} Hz")
        print(f"Antal røre-peaks: {analysis['num_peaks']}")
        
        # Assessment
        std_coefficient = analysis['std_gforce'] / analysis['mean_gforce']
        print(f"\nVARIATIONS KOEFFICIENT: {std_coefficient:.3f}")
        
        if std_coefficient > 0.1:
            print("✅ VURDERING: Høj variation indikerer god glutenudvikling")
        elif std_coefficient > 0.05:
            print("⚠️  VURDERING: Moderat variation - delvis glutenudvikling")
        else:
            print("❌ VURDERING: Lav variation - utilstrækkelig røring")
        
        print("\n📋 ANBEFALINGER:")
        total_minutes = analysis['total_time'] / 60
        if 8 <= total_minutes <= 15:
            print("✅ Røretid er optimal for sesamboller")
        elif total_minutes < 8:
            print("⚠️  Overvej længere røretid for bedre glutenudvikling")
        else:
            print("⚠️  Lang røretid - vær opmærksom på overrøring")
        
    except FileNotFoundError:
        print("❌ Fejl: Kunne ikke finde 'gForce_2025-05-22_12-09-20.csv'")
        print("Sørg for at filen er i samme mappe som dette script.")
    except Exception as e:
        print(f"❌ Fejl ved analyse: {str(e)}")

if __name__ == "__main__":
    main()