#!/usr/bin/env python3
"""
Fix Conda Environment for Oil Trading Dashboard
==============================================

This script installs the missing 'arch' package in your conda environment
so the full dashboard can run without dependency issues.
"""

import subprocess
import sys

def install_arch_package():
    """Install the arch package using conda or pip."""
    print("🔧 Installing 'arch' package for GARCH models...")
    
    try:
        # Try conda first
        result = subprocess.run([
            "conda", "install", "-y", "-c", "conda-forge", "arch"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully installed 'arch' package via conda")
            return True
        else:
            print("⚠️ Conda install failed, trying pip...")
            
            # Fallback to pip
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "arch"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Successfully installed 'arch' package via pip")
                return True
            else:
                print(f"❌ Failed to install 'arch': {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error installing 'arch' package: {e}")
        return False

def main():
    """Main installation process."""
    print("🛢️ Oil Trading Dashboard Environment Fix")
    print("=" * 45)
    
    if install_arch_package():
        print("\n✅ Environment fixed successfully!")
        print("You can now run the full dashboard:")
        print("   streamlit run streamlit_trading_dashboard.py")
    else:
        print("\n❌ Could not fix environment automatically.")
        print("Try running manually:")
        print("   conda install -c conda-forge arch")
        print("   or")
        print("   pip install arch")

if __name__ == "__main__":
    main()
