# Kubernetes Engine configuration

## gcloud configuration
Creating cluster

```
gcloud container clusters create iris-serving-cluster --num-nodes 1 --zone us-central1-f
```

Cluster Configuration

```
gcloud config set container/cluster iris-serving-cluster
```

```
gcloud container clusters get-credentials iris-serving-cluster --zone us-central1-f
```

## kubectl configuration

```
kubectl create -f kubernetes_config.yaml
```

### kubectl checks

``` 
kubectl get deployments
kubectl get pods
kubectl get services
kubectl describe service iris-service 
```

# Docker image
Docker image created to implement Iris queries across serving

[serving_iris](https://hub.docker.com/r/nbortolotti/serving_iris/)

*use version 2 tag.