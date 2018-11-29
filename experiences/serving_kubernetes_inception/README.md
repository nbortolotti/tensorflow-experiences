# Kubernetes Engine configuration

## gcloud configuration
Creating cluster

```
gcloud container clusters create inception-retrained-serving-cluster --num-nodes 1 --zone us-central1-f
```

Cluster Configuration

```
gcloud config set container/cluster inception-retrained-serving-cluster
```

```
gcloud container clusters get-credentials inception-retrained-serving-cluster --zone us-central1-f
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
kubectl describe service inception-retrained-service 
```

# Docker image
Docker image created to implement Iris queries across serving

[serving_iris](https://hub.docker.com/r/nbortolotti/serving_iris/)

*use version 2 tag.