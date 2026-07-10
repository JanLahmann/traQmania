# traQmania on a QuBins base image (core Qiskit + Aer + IBM Runtime, amd64 + arm64).
FROM ghcr.io/qubins/images:latest-small

COPY --chown=1000:100 . /opt/traqmania
RUN pip install --no-cache-dir /opt/traqmania

EXPOSE 8000
CMD ["python", "-m", "traqmania", "--host", "0.0.0.0", "--port", "8000"]
