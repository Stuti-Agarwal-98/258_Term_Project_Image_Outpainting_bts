FROM tensorflow/tfx:0.26.3
WORKDIR /pipeline
RUN pip install tensorflow_ranking
RUN pip install tensorflow_text
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"