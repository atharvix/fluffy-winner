# UNSW-NB15 Dataset Schema Reference & Preprocessing Specification

This document details the exact feature schema, data types, provenance, and zero-leakage preprocessing protocol for the **UNSW-NB15 Network Intrusion Detection Dataset** utilized in this project.

---

## 1. Dataset Provenance & Citation

- **Creators:** Nour Moustafa and Jill Slay.
- **Institution:** Australian Centre for Cyber Security (ACCS), University of New South Wales (UNSW Canberra at ADFA).
- **Publication Reference:**
  > Moustafa, Nour, and Jill Slay. *"UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."* 2015 Military Communications and Information Systems Conference (MilCIS). IEEE, 2015.
- **Official Source:** [UNSW Canberra Cybersecurity Research](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- **Release Partitions:**
  - `data/raw/UNSW_NB15_training-set.csv` (82,332 records, 45 columns, SHA256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`)
  - `data/raw/UNSW_NB15_testing-set.csv` (175,341 records, 45 columns, SHA256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`)

---

## 2. Target Semantics & Leakage Prevention Policy

### 2.1 Target Classes
- `0` $\rightarrow$ **`BENIGN`**: Legitimate, normal network traffic.
- `1` $\rightarrow$ **`ATTACK`**: Malicious intrusion flow across 9 categories (*Generic*, *Exploits*, *Fuzzers*, *DoS*, *Reconnaissance*, *Analysis*, *Backdoor*, *Shellcode*, *Worms*).

### 2.2 Excluded Identifier and Target Columns (Zero-Leakage Guarantee)
The following columns are strictly excluded from predictive model inputs:
1. `id`: Sequential row identifier. Using row indices introduces artificial ordering leakage.
2. `attack_cat`: Multi-class ground-truth label string. Revealing attack category trivializes binary detection.
3. `label`: Binary ground-truth target.

Neither `id`, `attack_cat`, nor `label` may ever be fed to the model or preprocessor.

---

## 3. Input Feature Schema (42 Total Features)

### 3.1 Categorical Flow Attributes (3 Features)
Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore', max_categories=10, sparse_output=False)` fitted strictly on training data.

| Feature Name | Type | Description | Common Values / Categories |
| :--- | :--- | :--- | :--- |
| `proto` | string | Transaction protocol | `tcp`, `udp`, `unas`, `arp`, `ospf`, `sctp`, `any`, `gre`, `rsvp`, `ipv6` (others binned into infrequent) |
| `service` | string | Application service | `-`, `dns`, `http`, `smtp`, `ftp`, `ftp-data`, `pop3`, `ssh`, `ssl`, `snmp`, `dhcp`, `radius`, `irc` |
| `state` | string | Connection state | `FIN`, `INT`, `CON`, `REQ`, `RST`, `ACC`, `CLO` |

### 3.2 Basic & Flow Features (15 Numerical Features)
Standardized via `StandardScaler()` fitted strictly on training data.

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `dur` | float | Record total duration (seconds) |
| `spkts` | integer | Source to destination packet count |
| `dpkts` | integer | Destination to source packet count |
| `sbytes` | integer | Source to destination transaction bytes |
| `dbytes` | integer | Destination to source transaction bytes |
| `rate` | float | Packet transmission rate (packets/sec) |
| `sttl` | integer | Source to destination time to live |
| `dttl` | integer | Destination to source time to live |
| `sload` | float | Source bits per second |
| `dload` | float | Destination bits per second |
| `sloss` | integer | Source packets retransmitted or dropped |
| `dloss` | integer | Destination packets dropped |
| `sinpkt` | float | Source interpacket arrival time (milliseconds) |
| `dinpkt` | float | Destination interpacket arrival time (milliseconds) |
| `sjit` | float | Source packet jitter (milliseconds) |
| `djit` | float | Destination packet jitter (milliseconds) |

### 3.3 TCP & Content Features (11 Numerical Features)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `swin` | integer | Source TCP window advertisement value |
| `stcpb` | integer | Source TCP base sequence number |
| `dtcpb` | integer | Destination TCP base sequence number |
| `dwin` | integer | Destination TCP window advertisement value |
| `tcprtt` | float | TCP connection setup round-trip time ($t_{\text{synack}} - t_{\text{syn}}$) |
| `synack` | float | TCP SYN to SYN-ACK setup time |
| `ackdat` | float | TCP SYN-ACK to ACK setup time |
| `smean` | integer | Mean packet size transmitted by source (bytes) |
| `dmean` | integer | Mean packet size transmitted by destination (bytes) |
| `trans_depth` | integer | Pipelined HTTP request depth |
| `response_body_len` | integer | HTTP response body length (bytes) |

### 3.4 Connection & Window Aggregation Features (13 Numerical Features)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `ct_srv_src` | integer | Connections to same service from source IP in window |
| `ct_state_ttl` | integer | Connections with same state and TTL combination |
| `ct_dst_ltm` | integer | Connections to same destination address in window |
| `ct_src_dport_ltm` | integer | Connections from source IP to same destination port |
| `ct_dst_sport_ltm` | integer | Connections from destination IP to same source port |
| `ct_dst_src_ltm` | integer | Connections between same source and destination IP pair |
| `is_ftp_login` | integer | 1 if FTP session was authenticated by credentials |
| `ct_ftp_cmd` | integer | Count of FTP commands transmitted in flow session |
| `ct_flw_http_mthd` | integer | Count of HTTP request methods (e.g. GET, POST) in flow |
| `ct_src_ltm` | integer | Connections from same source IP in last 100 connections |
| `ct_srv_dst` | integer | Connections to same service at destination IP in window |
| `is_sm_ips_ports` | integer | 1 if source and destination IP addresses and ports are equal |

---

## 4. Known Flaws and Mitigation Protocol

1. **Duplicate Records:**
   - In the raw 82,332-row training partition, 28,380 rows share identical feature values due to repeated network scanning patterns.
   - The dataset pipeline provides an explicit `deduplicate: bool` parameter. When deduplicated, 53,946 unique flows remain.
2. **Unseen Categories at Test/Inference Time:**
   - The test partition contains protocol categories not present in the training partition.
   - Mitigated via `OneHotEncoder(handle_unknown='ignore')`, ensuring unseen protocols map cleanly to all-zero categorical encodings without runtime failures.
3. **Partition-Only Preprocessing:**
   - Splitting occurs before fitting.
   - The fitted `ColumnTransformer` is serialized to `results/network_intrusion/preprocessor.joblib` and restored without re-fitting during inference.
