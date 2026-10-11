@'

\# Track 4 Phase 3 â€” Pipeline Runbook



\## Overview



Track 4 integrates synthetic data generation, evaluation, shared storage, and the Streamlit dashboard using Docker and local Kubernetes (Minikube).



The validated workflow is:



1\. Generate synthetic data using Gaussian Copula or CTGAN.

2\. Store generated CSVs and manifests on the `doppel-shared-pvc` PersistentVolumeClaim.

3\. Run the corresponding evaluation Job.

4\. Store evaluation JSON reports on the same PVC.

5\. View datasets, reports, metrics, and Pareto analysis in the dashboard.



\## Prerequisites



\- Docker Desktop

\- Minikube

\- `kubectl`

\- The Doppel repository and required cleaned dataset

\- Docker images built locally and available to Minikube



\## Deploy shared storage



```powershell

kubectl apply -f k8s/namespace/namespace.yaml

kubectl apply -f k8s/shared-storage-pvc.yaml

kubectl get pvc -n doppel
