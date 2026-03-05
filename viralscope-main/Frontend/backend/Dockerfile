FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
# Note: Railway builds from backend/ as root, so paths are relative to backend/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files (including llm/ subdirectory)
COPY . ./backend/

# Note: Model files (RandomForest.pkl, scaler.pkl) are optional
# They can be provided via volume mount in docker-compose.yml
# If not present, the app will use mock predictions

# Set PYTHONPATH to include app directory
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run the application using startup script
# This ensures PORT environment variable is properly expanded
CMD ["/app/start.sh"]

