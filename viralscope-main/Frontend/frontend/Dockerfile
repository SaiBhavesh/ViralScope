FROM node:18-alpine

WORKDIR /app

# Accept build-time arguments for environment variables
# These will be passed from Railway during build
ARG NEXT_PUBLIC_BACKEND_URL
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci || npm install

# Copy application code
COPY . .

# Build the application (NEXT_PUBLIC_BACKEND_URL will be available here)
RUN npm run build

# Expose port (Railway will set PORT env var)
EXPOSE 3000

# Run the application
# Use start:railway for Railway (respects PORT env var)
# Use start for local (forces port 3000)
CMD ["npm", "run", "start:railway"]
