# One image, not two. frontend/app/api/process-video/route.ts spawns the Python scripts
# as child processes on a shared filesystem, so splitting frontend and backend into
# separate services would mean rewriting that route to speak HTTP instead of spawn().
#
# CUDA comes from the pip torch wheels (they bundle the runtime); only the driver comes
# from the host via nvidia-container-toolkit. That is why this is a plain node base
# rather than an nvidia/cuda one.
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev \
      build-essential git \
      libgl1 libglib2.0-0 libgomp1 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Debian marks its python as externally managed (PEP 668), so pip needs a venv.
# It also gives PYTHON_BIN a stable path for the spawn() in route.ts.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHON_BIN=/opt/venv/bin/python

WORKDIR /app

# Python deps first: slowest layer, changes least often.
# requirements.txt pins torch==2.9.1+cu130, which only exists on the PyTorch index.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r backend/requirements.txt

# Node deps next, before the source copy, so editing code does not re-run npm ci.
COPY frontend/package.json frontend/package-lock.json frontend/
WORKDIR /app/frontend
RUN npm ci

WORKDIR /app
COPY . .

WORKDIR /app/frontend
RUN npm run build

# No display in a container: step2's draw_geometries() would block forever.
ENV CONSTRUCT_SHOW_VIZ=0
# Weights are 6.3 GB. Mount this, never bake them into the image.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 3000
CMD ["npm", "run", "start"]
