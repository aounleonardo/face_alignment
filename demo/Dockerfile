FROM continuumio/miniconda3:4.12.0

RUN apt update && apt upgrade -y && \
    apt install -y \
        libgl1 \
        lsof \ 
        tmux \
        vim

WORKDIR /face_alignment

COPY ../requirements.txt .
RUN pip install -r requirements.txt && \
    pip install scikit-image

COPY ../setup.py .
COPY ../ibug ibug 
RUN pip install -e .

COPY demo/main.py .

RUN conda install jupyterlab matplotlib ipywidgets

COPY demo/notebook.ipynb .

CMD [ \
    "tmux", \ 
    "new-session", "/bin/bash", ";", \
    "new-window", "jupyter-lab --ip=0.0.0.0 --no-browser --allow-root" \
]
