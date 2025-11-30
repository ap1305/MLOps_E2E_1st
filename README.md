# MLOps_E2E_1st
1st E2E Project
Wine quality :- Dataset
1. select best model using gridsearchCV.scaling..
2.save the model.
3. create a new python file load the model in it . and make the use of model using fastapi.
later part :- ( 1st add MLflow then add DVC )
Use GitHub action :-
Containerize it :- We will use deployment (k8s ) with 2 replicas.
## Monitoring
####when Everything will be done :- then same model we will deploy using streamlit and flask
Stage 2 :- make a separate config file to maintain values that can be changed .


 week three :- Terraform


#Scaler
-> joblib.dump(scaler, 'scaler.pkl')
-> scaler = joblib.load('scaler.pkl')
-> X_scaled = scaler.transform(X_new)  # Apply to new data


thrid parties neeed on ec2:-
docker
docker compose

How to install third parties:-
docker installtion :- 
sudo yum install docker

docker compose installation:-
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
