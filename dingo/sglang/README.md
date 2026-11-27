<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## Restore routing lanes on worker startup

SGLang workers accept `--model-taints` (environment: `DYN_MODEL_TAINTS`). The
value is a JSON array of routing taints, a JSON object mapping exact Pod names
to arrays, or `@/path/to/taints.json`. CLI takes precedence over the environment.
Workers validate this configuration before creating the engine and include the
resolved taints in their first model registration. No post-readiness POST is
needed to restore the configured lane after a restart or Pod recreation.

For a workload whose replicas all serve one lane, set:

```bash
--model-taints '["lane/GL"]'
```

For three lanes in one StatefulSet, persist the assignments in a ConfigMap.
Replace the names below with the actual StatefulSet Pod names:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pool-model-taints
data:
  prefill.json: |
    {
      "pool-prefill-0": ["lane/S"],
      "pool-prefill-1": ["lane/GL"],
      "pool-prefill-2": ["lane/BL"]
    }
  decode.json: |
    ["lane/S", "lane/GL", "lane/BL"]
```

Add the following fields to the **existing** prefill Pod template, retaining
its resource, scheduling, storage and other environment settings:

```yaml
spec:
  containers:
    - name: worker  # Use the existing worker container name.
      env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: DYN_MODEL_TAINTS
          value: "@/etc/dingo/model-taints/prefill.json"
      volumeMounts:
        - name: model-taints
          mountPath: /etc/dingo/model-taints
          readOnly: true
  volumes:
    - name: model-taints
      configMap:
        name: pool-model-taints
```

Shared decode workers use the same mount and set `DYN_MODEL_TAINTS` to
`@/etc/dingo/model-taints/decode.json`, advertising all three lane taints.
Nexus still sends `nvext.routing_constraints.required_taints` to the shared
Router; for example, `["lane/S"]` selects the short lane. These are model
routing taints, separate from Kubernetes node taints and Pod labels. Requests
without required taints can still reach any otherwise eligible worker.

A per-Pod map requires `POD_NAME` and a matching entry; missing files, missing
assignments, malformed values and the reserved `dynamo.topology/` prefix fail
startup. Add assignments before scaling out. An explicit `[]` registers no
caller-managed taints; omitting the setting preserves existing behavior.

This configuration is read at startup, without a file or Pod-label watcher.
`POST /engine/update/model_taints` still replaces the running worker's
caller-managed taints without restarting it, but does not persist the change.
To retain a lane change after recreation, also update the ConfigMap source;
the next worker start reads the updated assignment. A ConfigMap edit alone
does not move a running worker. Topology-generated taints remain independent.

## KVCR-owned inventory

KVCR inventory publication is opt-in for the in-process `python -m dynamo.sglang`
HiCacheStorage and direct linker paths. It requires a matching SGLang backend that
reports the `kvcr_inventory` capability and engine-lifetime cache-owner metadata.

For HiCacheStorage, configure the KVCR backend with
`"publish_inventory": true`, remote hints enabled, and an explicit reachable
`control_advertise_host`.

For the experimental direct linker, use TP1 and configure the worker with:

```bash
--tp-size 1 \
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend kvcr \
--hicache-storage-backend-extra-config '{"publish_inventory": true, "kvcr_service_socket_path": null, "advertise_host": "192.0.2.10"}'
```

Replace `192.0.2.10` with a non-loopback address reachable by peers. Set distinct
control and NIXL ports for colocated workers. Do not use HiCacheStorage options
such as `control_advertise_host` in the linker configuration. See the
[experimental linker quick start](https://github.com/ai-dynamo/kvcr/pull/129) for
a complete example.

For either path, enable KV events with a ZMQ publisher and `"format": "dynamo"`.
Start a state-agent host in the workers' namespace before starting the workers:

```bash
DYN_NAMESPACE=<worker-namespace> python -m dynamo.kv_state_agent
```

The worker reads the raw-event endpoint, incarnation-specific topic, cache owner,
and peer-control endpoint from SGLang startup metadata. It attaches before model
registration and uses `state_agent_v2` instead of the legacy event publisher.
The router can then distinguish framework GPU/HiCache-host residency from actual
KVCR-owned DRAM residency, even though both host caches use CPU memory.

Upgrade the Dynamo frontend, worker, and state-agent host together when enabling
this mode. Existing deployments without the capability keep the legacy path;
vLLM attachment descriptors without an ingress protocol keep their existing
default. The new attachment requires a single-host LLM engine, PP=1, and one
tokenizer worker; the direct linker additionally requires TP1 and DP1. Incomplete
source metadata is rejected at startup.

This is a live process-local inventory, not durable recovery: detaching the
engine clears its KVCR owner, and a detected raw-event sequence gap withdraws
readiness until the worker is restarted. Router hints remain advisory; storage
lookup and completed delivery still determine usable pages.
