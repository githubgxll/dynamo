# Persistent Dingo GitHub Actions runners

This directory deploys persistent repository-level GitHub Actions runners:

| Runner | Repository | Namespace | Manifest | Persistent volume |
| --- | --- | --- | --- | --- |
| `dingo-runner-k8s-1` | `zhaoxianhua/dynamo` | `dingo-runner` | `dingo-runner-k8s-1.yaml` | `dingo-runner-k8s-1-data` |
| `dingo-gxl-runner` | `githubgxll/dynamo` | `dingo-runner` | `dingo-gxl-runner.yaml` | `dingo-gxl-runner-data` |

GitHub adds the standard `self-hosted`, `linux`, and `x64` labels. Both
deployments add `hd-04`, `dingo`, and `kubernetes`; `dingo-gxl-runner` also
adds `gxl`. Workflows require the `dingo` label so unrelated self-hosted
runners cannot claim these jobs. Both runners are restricted to the
`hd04-cci-k8s-worker-1` through `hd04-cci-k8s-worker-3` worker nodes.

The runner and Docker Engine execute in one privileged container. Runner
configuration, workspaces, Docker layers, and named Buildx builders are stored
on a 500 GiB RWO Ceph RBD volume so `keep_buildkit_state: true` continues to
work across Pod recreation. The runner home plus Rustup and Cargo state are
persistent as well, and Rust toolchains use the rsproxy distribution mirror.
Cargo uses its sparse registry protocol and the Git CLI for Git dependencies.
GitHub, Docker Hub, package managers, and Docker build steps use the hd-04
egress proxy at `http://10.201.136.68:1080`. Cluster networks and the hd-04
private registry are excluded through `NO_PROXY`; GitHub Actions broker
endpoints also stay direct so job delivery does not depend on proxy long-poll
support.
The Swagger UI archive required by Rust builds is checksum-pinned on the PVC
and exposed through `SWAGGER_UI_DOWNLOAD_URL` as a local file.
The Kubernetes ServiceAccount token is not mounted.
The bootstrap pins GitHub Actions Runner 2.337.0 and Docker Buildx 0.37.0 with
SHA256 verification, and installs the Python renderer dependencies required by
the current workflow. It uses the Huawei Cloud Ubuntu mirror over HTTPS with
retry enabled because the base image's HTTP mirror can briefly serve
inconsistent indexes or cached packages during synchronization.

## One-time registration

Create the repository-specific runner registration token at one of:

`https://github.com/zhaoxianhua/dynamo/settings/actions/runners/new`

`https://github.com/githubgxll/dynamo/settings/actions/runners/new`

The token is short-lived. Enter it without echoing it into the terminal:

```bash
tsh ssh --user=zhaoli@zetyun.com root@hd04-cci-k8s-master-1
read -rsp 'GitHub runner registration token: ' RUNNER_TOKEN; echo
kubectl -n dingo-runner create secret generic dingo-runner-k8s-1-registration \
  --from-literal=token="${RUNNER_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
unset RUNNER_TOKEN
```

For `githubgxll/dynamo`, use the same procedure with the independent Secret:

```bash
read -rsp 'GitHub runner registration token: ' RUNNER_TOKEN; echo
kubectl -n dingo-runner create secret generic dingo-gxl-runner-registration \
  --from-literal=token="${RUNNER_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
unset RUNNER_TOKEN
```

Apply the deployment from the master host:

```bash
kubectl apply -f /root/zhaoli/dingo-runner-k8s-1/dingo-runner-k8s-1.yaml
kubectl apply -f /root/zhaoli/dingo-gxl-runner/dingo-gxl-runner.yaml
```

Verify registration and Docker readiness:

```bash
kubectl -n dingo-runner get pod -l app.kubernetes.io/name=dingo-runner-k8s-1 -o wide
kubectl -n dingo-runner logs -f deployment/dingo-runner-k8s-1
kubectl -n dingo-runner get pod -l app.kubernetes.io/name=dingo-gxl-runner -o wide
kubectl -n dingo-runner logs -f deployment/dingo-gxl-runner
```

After the log reports `Runner successfully added`, replace the consumed
short-lived token with a non-sensitive value. The Secret must remain because
the Pod mounts it, but normal restarts use the `.runner` file on the PVC:

```bash
kubectl -n dingo-runner create secret generic dingo-runner-k8s-1-registration \
  --from-literal=token=registration-complete \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n dingo-runner create secret generic dingo-gxl-runner-registration \
  --from-literal=token=registration-complete \
  --dry-run=client -o yaml | kubectl apply -f -
```

The current workflows route both generic tests and Runtime image builds using
the `self-hosted,linux,x64,dingo` labels. Repository-level registration keeps
the two runners isolated even though they share the routing labels. The
current image configuration publishes to the hd-04 registry.

## Recovery

The one-time registration token is only needed while `.runner` is absent on
the persistent volume. Normal Pod restarts reuse the saved registration.

If the runner is removed in GitHub or the PVC is replaced, create a fresh
registration token, update the Secret, remove the stale runner configuration
from the PVC, and restart the Deployment.
