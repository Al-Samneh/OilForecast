#!/usr/bin/env python3
"""
Quick Setup and Launch Script for Oil Trading Dashboard
======================================================

This script helps you quickly set up and launch the interactive trading dashboard.
It will generate fresh trading data and launch the Streamlit interface.

Usage:
    python run_dashboard.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import streamlit
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        return True

def generate_trading_data():
    """Generate fresh trading data by running the test script."""
    print("🔄 Generating fresh trading data...")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_2024_2025_performance.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Trading data generated successfully")
            return True
        else:
            print(f"❌ Error generating data: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏱️ Data generation timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_data_exists():
    """Check if trading data files already exist."""
    output_dir = Path("outputs/2024_2025_test")
    required_files = [
        "predictions_2024_2025.csv",
        "signals_2024_2025.csv", 
        "test_results_summary.json"
    ]
    
    if output_dir.exists():
        existing_files = [f for f in required_files if (output_dir / f).exists()]
        if len(existing_files) == len(required_files):
            print("✅ Trading data files found")
            return True
    
    print("📊 No trading data found - need to generate fresh data")
    return False

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching trading dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "simple_dashboard.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main setup and launch sequence."""
    print("🛢️ Oil Trading Dashboard Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Check if data exists
    if not check_data_exists():
        # Non-interactive: attempt to generate trading data automatically
        if not generate_trading_data():
            print("❌ Failed to generate trading data")
            return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
