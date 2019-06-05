FROM python:3.6


ADD ./recognize_service /home
#ADD ./recognize_service/requirements.txt /home
WORKDIR /home
RUN cd /home
RUN ls /home
RUN apt-get update -y
RUN apt-get install libgtk2.0-dev -y
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]