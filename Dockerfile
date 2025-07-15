FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
# Set working directory
WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y git build-essential cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install flask flask-cors

RUN chmod +x ./backend/start.sh

ENTRYPOINT ["bash", "./backend/start.sh"]

CMD ["5000"]