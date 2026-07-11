SOUL : Soft Optimization through Unified Trajectory Alignment against Free-Rider Attacks in Non-IID Federated Learning


📌 Overview

SOUL is a soft optimization through unified trajectory alignment framework against free-rider attacks in Non-IID Federated Learning. 

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
│   │   ├── soul.yaml
│   ├── base.yaml
│
│
├── soul.py
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
