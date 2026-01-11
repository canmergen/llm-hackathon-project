#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run src/streamlit_compliance_viewer.py --server.port 8501
