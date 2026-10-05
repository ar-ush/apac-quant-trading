FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# keys come from the environment / --env-file, never from the image
# default: DRY-RUN. For real orders:  docker run --env-file .env <image> python -m bot.main --live
CMD ["python", "-m", "bot.main"]
