UNFLIP: A Federated Unlearning Framework for Label Flipping Resilience under IID and Non-IID Data


📌 Overview

UNFLIP is a federated unlearning framework for label flipping resilience to keep the model working even if anywhere from 0\% to 100\% of the clients are malicious. 

🎯 Key Objectives

Maintain model stability under high adversarial presence.

Distinguish malicious updates from diverse non-IID data.

Align local updates using a trusted shadow manifold.

Preserve accuracy with a minimal, carefully curated root dataset.


```bash
UNFLIP/
│
├── conf/
│   ├── strategy/
│   │   ├── ntn.yaml
│   ├── base.yaml
│
│
├── ntn.py
├── client.py
├── counter.txt
├── dataset.py
├── model.py
├── server.py
├── catforget_model.py
├── main.py
│
│
└── README.md
```

Running a Sample Experiment: 

python main.py
