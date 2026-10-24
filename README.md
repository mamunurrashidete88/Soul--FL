# Artifact for  *The Sleeper's Influence: Limiting Delayed Sybil Poisoning in Federated Learning*. 

## Installation and quick functional check

Python 3.10+ and a compatible PyTorch installation are required for training. The protocol tests do not require image or recommendation data.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Datasets and image experiments


```sh
python run_simulation.py --dataset cifar10 --seed 11 --rounds 200 --output experiments/logs/cifar11.csv
python run_simulation.py --dataset femnist --data /path/to/leaf/femnist --seed 11 --output experiments/logs/femnist11.csv
python run_simulation.py --dataset cifar10 --seed 11 --reused-data-keys 100 --output experiments/logs/reused100_seed11.csv
```


## MovieLens-1M


```sh
python run_recommendation.py --data /path/to/ml-1m --seed 11 --keys 40 --attack injection --method soul --output experiments/logs/web11.csv
python run_recommendation.py --data /path/to/ml-1m --seed 11 --keys 40 --attack optimized --method screen-fixed --intermit --output experiments/logs/web_control.csv
```

## Contract build and proof integration

Node.js and npm are required for the Solidity prototype. 

```sh
npm install
npm run compile
npm test
```



## Repository structure

```text
Soul-FL-main/
├── README.md
└── Soul-FL-Artifact/
    ├── README.md                 Setup, scope, and reproduction notes
    ├── requirements.txt          Python dependencies
    ├── package.json              Hardhat and Ethers dependencies
    ├── hardhat.config.js         Contract build configuration
    ├── config.py                 Image-run parameters
    ├── data_loader.py            CIFAR-10 and LEAF FEMNIST loaders and partitioning
    ├── run_simulation.py         Image training and reused-data key sweep
    ├── run_recommendation.py     MovieLens-1M transfer experiment
    ├── core/
    │   ├── aggregation.py        Pre-credit balance-weighted aggregation
    │   ├── client.py             Local image training and delayed label flip
    │   ├── models.py             Image architectures
    │   └── server.py             Calibration, screening, and evaluation
    ├── security/
    │   ├── blockchain_sim.py     Signed-voucher off-chain ledger
    │   ├── cvae.py               Anchor-conditioned detector
    │   ├── projector.py          Fitted incremental PCA
    │   ├── trust_engine.py       Integer decay and aggregation weights
    │   └── zk.py                 Local enrollment-relation checker (not ZK)
    ├── contracts/
    │   ├── SoulFLToken.sol       Voucher and balance contract
    │   └── RejectingVerifier.sol Negative-test verifier
    ├── scripts/
    │   ├── deploy_and_test.js    Deployment requiring an external verifier
    │   ├── smoke_protocol.py     Dependency-light protocol checks
    │   └── summarize_image.py    Summary from new image-run CSVs
    ├── test/negative_enrollment.js
    └── tests/test_protocol.py
```



## License and citation

Provided for academic peer review. No separate software license is included in this archive.
