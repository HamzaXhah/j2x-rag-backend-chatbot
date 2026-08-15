# Production build (multi-stage) with uv
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install build dependencies including PostgreSQL dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install dependencies with uv into system Python (fast and reliable)
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --no-cache torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --no-cache -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with fixed UID/GID and no shell access
RUN addgroup --system --gid 10001 app && \
    adduser --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app . .

# The named model volume must be writable by the non-root runtime user.
RUN mkdir -p /app/models && chown -R app:app /app/models

# Switch to non-root user
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

ENTRYPOINT ["sh", "./docker-entrypoint.sh"]


# Development build with uv
FROM python:3.11-slim AS development

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install with uv into system Python (fast!)
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --no-cache torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/raw /app/data/processed

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

ENTRYPOINT ["sh", "./docker-entrypoint.sh"]
CMD ["--reload"]
