import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
import secrets
from config import AUTH_CONFIG, SECURITY_CONFIG, FEATURES

class AuthManager:
    def __init__(self, users_file=None):
        self.users_file = users_file or AUTH_CONFIG["users_file"]
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            # Default admin user
            self.users = {
                AUTH_CONFIG["default_admin_username"]: {
                    "password": self.hash_password(AUTH_CONFIG["default_admin_password"]),
                    "role": "admin",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None
                }
            }
            self.save_users()
    
    def save_users(self):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    def register_user(self, username, password, role="user"):
        """Register a new user"""
        if username in self.users:
            return False, "Username already exists"
        
        if len(password) < AUTH_CONFIG["password_min_length"]:
            return False, f"Password must be at least {AUTH_CONFIG['password_min_length']} characters long"
        
        self.users[username] = {
            "password": self.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        self.save_users()
        return True, "User registered successfully"
    
    def login_user(self, username, password):
        """Login user and return success status"""
        if username not in self.users:
            return False, "Invalid username or password"
        
        if not self.verify_password(password, self.users[username]["password"]):
            return False, "Invalid username or password"
        
        # Update last login
        self.users[username]["last_login"] = datetime.now().isoformat()
        self.save_users()
        
        return True, "Login successful"
    
    def get_user_role(self, username):
        """Get user role"""
        if username in self.users:
            return self.users[username]["role"]
        return None
    
    def delete_user(self, username):
        """Delete a user (admin only)"""
        if username in self.users:
            del self.users[username]
            self.save_users()
            return True, "User deleted successfully"
        return False, "User not found"

def init_auth():
    """Initialize authentication in session state"""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None

def login_page():
    """Display login page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 InsurAI Login</h1>
        <h2>Secure Access to Insurance Agentic Workflow</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    border-radius: 20px; 
                    padding: 2rem; 
                    border: 1px solid rgba(102, 126, 234, 0.3);
                    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);">
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            st.markdown("<h3 style='text-align: center; color: #00ff88; margin-bottom: 2rem;'>Sign In</h3>", unsafe_allow_html=True)
            
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            # User type selection
            user_type = st.radio(
                "👥 User Type",
                ["User", "Admin"],
                horizontal=True,
                help="Select your user type for login"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col2:
                if FEATURES["enable_user_registration"]:
                    register_button = st.form_submit_button("📝 Register", use_container_width=True)
                else:
                    register_button = False
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Handle login
        if login_button and username and password:
            success, message = st.session_state.auth_manager.login_user(username, password)
            if success:
                # Check if user type matches the selected role
                actual_role = st.session_state.auth_manager.get_user_role(username)
                selected_role = user_type.lower()
                
                if actual_role == selected_role:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = actual_role
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error(f"Access denied. This account is registered as '{actual_role}', not '{selected_role}'.")
            else:
                st.error(message)
        
        # Handle registration
        if register_button and username and password and FEATURES["enable_user_registration"]:
            success, message = st.session_state.auth_manager.register_user(username, password, role=user_type.lower())
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Show default credentials
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%);
                    border-radius: 15px; 
                    padding: 1.5rem; 
                    margin: 2rem 0;
                    border: 1px solid rgba(255, 152, 0, 0.3);">
            <h4 style='color: #ff9800; margin-bottom: 1rem;'>🔑 Default Credentials</h4>
            <p style='margin: 0.5rem 0;'><strong>Username:</strong> {AUTH_CONFIG['default_admin_username']}</p>
            <p style='margin: 0.5rem 0;'><strong>Password:</strong> {AUTH_CONFIG['default_admin_password']}</p>
            <p style='margin: 1rem 0 0 0; font-size: 0.8; opacity: 0.8;'>Use these credentials to login or register a new account.</p>
        </div>
        """, unsafe_allow_html=True)

def logout_button():
    """Display logout button in sidebar"""
    if st.session_state.authenticated:
        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3>👤 User Profile</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(139, 195, 74, 0.1) 100%);
                    border-radius: 10px; 
                    padding: 1rem; 
                    margin: 0.5rem 0;
                    border: 1px solid rgba(76, 175, 80, 0.3);">
            <p style="margin: 0.2rem 0; color: white;"><strong>User:</strong> {st.session_state.username}</p>
            <p style="margin: 0.2rem 0; color: white;"><strong>Role:</strong> {st.session_state.user_role}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.rerun()

def require_auth():
    """Decorator to require authentication for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.authenticated:
                st.error("Please login to access this feature.")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

def admin_only():
    """Decorator to require admin role for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.authenticated:
                st.error("Please login to access this feature.")
                return None
            if st.session_state.user_role != "admin":
                st.error("Admin access required for this feature.")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

def admin_panel():
    """Admin panel for user management"""
    if st.session_state.user_role != "admin" or not FEATURES["enable_admin_panel"]:
        return
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>⚙️ Admin Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("👥 User Management"):
        # Show all users
        st.markdown("**Registered Users:**")
        for username, user_data in st.session_state.auth_manager.users.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {username} ({user_data['role']})")
            with col2:
                if username != "admin" and st.button("🗑️", key=f"del_{username}"):
                    success, message = st.session_state.auth_manager.delete_user(username)
                    if success:
                        st.success("User deleted!")
                        st.rerun()
                    else:
                        st.error(message)
        
        # Add new user
        st.markdown("---")
        st.markdown("**Add New User:**")
        new_username = st.text_input("Username", key="new_user")
        new_password = st.text_input("Password", type="password", key="new_pass")
        new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
        
        if st.button("➕ Add User"):
            if new_username and new_password:
                success, message = st.session_state.auth_manager.register_user(new_username, new_password, new_role)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill all fields") 