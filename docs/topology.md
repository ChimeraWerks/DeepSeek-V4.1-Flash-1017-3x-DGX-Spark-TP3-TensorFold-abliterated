# Three Spark topology

Run one TensorFold rank per NVIDIA DGX Spark GB10: three systems with 128 GB unified memory each. Connect every pair directly over RoCE.

```text
                    spark1
             head and engine rank 0
                 /           \
              RoCE           RoCE
               /               \
       spark2 ------- RoCE ------- spark3
       rank 1                      rank 2
```

| Node | Role | Storage |
|---|---|---|
| First in `SPARK_HOSTS` | Head, engine rank 0, serving API | Weights, local rank split, runtime files, logs and 256 GiB NVMe spill |
| Second in `SPARK_HOSTS` | Engine rank 1 | Weights, local rank split, runtime files and logs |
| Third in `SPARK_HOSTS` | Engine rank 2 | Weights, local rank split, runtime files and logs |

Reserve about 440 GB free per Spark: 274 GB for the weights, the local rank file (about 93 GB), 67.6 GB of Engram rows and the kernel kit ([runtime assets](runtime-assets.md)). Add the head's 256 GiB spill and operational files separately.

`SPARK_ROOT` holds `weights/`, `ranks/`, `bin/`, `kit/`, `spill/`, `logs/`, `guard/`, `service/` and `api/`. See [cluster.env.example](../cluster.env.example) for optional path overrides.

## SSH and RoCE

- `SPARK_HOSTS`: SSH names in rank order, head first.
- `SPARK_USER`: common SSH user.
- `SSH_KEY`: local SSH identity for the launch checkout to reach the Sparks.
- `HEAD_SSH_KEY`: identity on the head for head-to-worker checks.

`MASTER_ADDR` is the head's RoCE rendezvous address; `NCCL_IB_ADDR_RANGE` selects the RoCE subnet. SSH host names identify SSH connections, not the rendezvous.

The measured cluster used ConnectX-7 and four RDMA devices per node. Set `NCCL_IB_HCA`, `TF_RDMA_DEVICES`, `NCCL_SOCKET_IFNAME`, `TF_RDMA_GID_INDEX` and `RDMA_PEER_MASKS` to match the interfaces and cabling. Production enables owned-row prefill and direct all-gathers; see [tuning](tuning.md).

## Private service access

The head runs the native engine on `ENGINE_PORT` 8003 and the serving API on `API_PORT` 18002. Both bind loopback by default. Authenticate clients at the serving API, not the engine. Keep both ports private; connect remote API clients through SSH forwarding or an authenticated private reverse proxy.

Run [Preflight and launch](reproduction.md) on the configured cluster. Provision networks and firmware separately. Read [security](../SECURITY.md) before connecting clients.
