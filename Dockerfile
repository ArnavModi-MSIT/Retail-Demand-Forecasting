# ── Build stage ────────────────────────────────────────────────────────────
# Compiles any packages needing build-essential, then discarded — keeps
# the final image lean for a memory-constrained EC2 micro instance.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# curl needed only for the HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the app
RUN useradd --create-home --shell /bin/bash appuser

# Bring in pre-built Python packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

COPY . .

# Hand ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Make sure pip-installed console scripts are on PATH for the new user
ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]