# FedAD-Bench
Benchmark for Federated Unsupervised Anomaly Detection from the paper: 

### ***[FedAD-Bench: A Unified Benchmark for Federated Unsupervised Anomaly Detection in Tabular Data](https://arxiv.org/abs/2408.04442)***

___

### **Dependencies**

To install dependencies you can run the following command:

` pip install -r requirements.txt `

___

### **Example**

Here is an example how to run an experiment with *three* clients using a vanilla *Autoencoder* on the *Arrhythmia* dataset:

` python main.py experiment=example-exp model_name=AE model=dae dataset=arrhythmia batch_size=16 n_epochs=2 num_rounds=2 model.latent_dim=3  num_clients=3`

___

### **Visualization**

You can visualize the results of an experiment or multiple experiments using the script `review_results.py`
