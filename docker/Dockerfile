FROM pytorch/pytorch

# Copy
COPY .. /kcg-ml-elm
WORKDIR /kcg-ml-elm

RUN pip3 install -r ./requirements-docker.txt

WORKDIR /kcg-ml-elm
CMD ["python3", "run-all-notebooks.py"]