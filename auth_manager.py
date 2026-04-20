import streamlit as st

class AuthManager:
    def __init__(self):
        # user_db format: {"username": {"password": "...", "api_key": "..."}}
        if "user_db" not in st.session_state:
            st.session_state.user_db = {} 
        if "authenticated_user" not in st.session_state:
            st.session_state.authenticated_user = None

    def login(self, username, password):
        user = st.session_state.user_db.get(username)
        if user and user["password"] == password:
            st.session_state.authenticated_user = username
            return True
        return False

    def register(self, username, password, api_key):
        if username and password and api_key:
            st.session_state.user_db[username] = {
                "password": password,
                "api_key": api_key
            }
            return True
        return False

    def get_api_key(self):
        user = st.session_state.authenticated_user
        return st.session_state.user_db[user]["api_key"] if user else None

    def logout(self):
        st.session_state.authenticated_user = None