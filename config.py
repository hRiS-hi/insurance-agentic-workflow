# Authentication Configuration
AUTH_CONFIG = {
    "users_file": "users.json",
    "session_timeout": 3600,  # 1 hour in seconds
    "max_login_attempts": 3,
    "password_min_length": 6,
    "default_admin_username": "admin",
    "default_admin_password": "admin123"
}

# App Configuration
APP_CONFIG = {
    "page_title": "InsurAI - Insurance Agentic Workflow",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_rate_limiting": True,
    "rate_limit_requests": 100,  # requests per hour
    "enable_session_management": True,
    "enable_password_reset": False,  # Set to True if you want password reset functionality
}

# Feature Flags
FEATURES = {
    "enable_user_registration": True,
    "enable_admin_panel": True,
    "enable_file_upload": True,
    "enable_chat": True,
    "enable_data_export": True
} 