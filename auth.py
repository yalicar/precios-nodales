import streamlit as st

def check_auth():
    return st.session_state.get("authenticated", False)
