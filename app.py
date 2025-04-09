"""Yes It Works Documentation Helper - Main application entry point.

A Streamlit application that provides an interactive chat interface to help users
understand and navigate the Yes It Works tool's features and capabilities.
"""
from src.ui.streamlit_app import YesItWorksApp

def main():
    # Initialize and run the app
    app = YesItWorksApp()
    app.run()

if __name__ == "__main__":
    main()
