FROM pytorch/pytorch

# Copy
COPY . /kcg-ml-elm
WORKDIR /kcg-ml-elm

RUN pip3 install -r ./notebooks/requirements-docker.txt

# delete for now
RUN rm -R notebooks/dataset_download

WORKDIR /kcg-ml-elm/notebooks
CMD ["python3", "run-all-notebooks.py"]