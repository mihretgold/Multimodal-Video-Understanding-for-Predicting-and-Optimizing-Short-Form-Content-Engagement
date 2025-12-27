# Procfile for Render/Heroku deployment
# ======================================
# Note: render.yaml is preferred for Render, but this works as fallback

web: cd backend && gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 "app.app:create_app()"

