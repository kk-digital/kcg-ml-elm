FROM pytorch/pytorch

# Copy
COPY . /kcg-ml-elm
WORKDIR /kcg-ml-elm

RUN pip3 install -r ./notebooks/requirements-docker.txt

WORKDIR /kcg-ml-elm/notebooks
CMD ["python3", "run-all-notebooks.py"]