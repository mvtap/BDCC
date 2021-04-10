FROM gcr.io/google-appengine/python
RUN virtualenv /env -p python3
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ENV GOOGLE_CLOUD_PROJECT=bdcc-project1-309010
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ADD . /app
WORKDIR /app

CMD [ "python3" , "main.py", "--host=0.0.0.0"]