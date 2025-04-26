
FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y libgl1-mesa-glx


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501


CMD ["streamlit", "run", "MatID_APP.py", "--server.port=8501", "--server.address=0.0.0.0"]
