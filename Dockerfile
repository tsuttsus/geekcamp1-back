FROM python:3.9

RUN mkdir /code
WORKDIR /code

RUN pip install --upgrade pip --no-cache-dir
ADD requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

#ADD entrypoint.sh /code/entrypoint.sh
ADD . /code

EXPOSE 5000
ENV PYTHONPATH "${PYTHONPATH}:/code/"
ENV FLASK_APP "/code/run.py"
#CMD ["/code/entrypoint.sh"]
CMD ["python","run.py"]