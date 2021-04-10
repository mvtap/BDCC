FROM gcr.io/google-appengine/python
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ADD requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN ls -l
RUN pip3 install -r /app/requirements.txt
COPY . .
CMD ["flask", "run"]
