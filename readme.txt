For LR-DDoS attack detection we do the following setup:

>CICIDS2017 dataset (preprocess the dataset and label each anomalies, and separate ddos attack from orginal dataset)

>for simulating SDN network we using mininet in linux and install ryu controller in mininet

>next to create a topology in mininet using one controller, 2/3 switch and 2 pair of host for each switch

>command for running mininet: sudo mn --controller=remote,ip=127.0.0.1,port=6633 --topo single,3

>command for running ryu controller: ryu-manager my_ryu_app.py

>command for running host: mininet> xterm h1 h2 h3 (act one host as attacker and another as victim)

>in one host run lrddos attack



///////////////////////////////
Create a Ryu Application File:

>Open a text editor or IDE.
>Create a new Python file (e.g., anomaly_detection_app.py) inside the ~/ryu_apps directory.
>Paste the provided Ryu application code into this file and save it.
///////////////////////////////

>download CICIDS2017 dataset.csv(300mb)
> divide the dataset for training the model and testing .It is recommended to use 80% for training and 20% for testing.