# Dingo runtime image CI

The workflow in `workflows/dingo-router-ci.yml` runs whenever a commit is
pushed to `DingoRouter-base`. All jobs use a local Linux x64 self-hosted
runner. By default it builds one standalone Dynamo runtime image, which does
not inherit or install an inference-framework runtime. The existing vLLM and
SGLang runtime builds remain available as manual overrides.

## Image configuration

Edit `dingo-images.json` to control the builds:

- Supported `framework` values are `dynamo`, `vllm`, and `sglang`.
- Set the image's `enabled` value to `true` or `false`.
- `repository` controls the image repository name.
- `tag_prefix` controls the part of the tag before the commit SHA.
- `commit_sha_length` controls how many commit SHA characters are appended.
- `platform` is currently `linux/amd64` and `cuda_version` is currently `13.0`.
- `keep_buildkit_state` preserves cross-run BuildKit caches when set to
  `true`. It is disabled by default to limit disk use.

With the checked-in defaults, commit `0123456789abcdef...` publishes:

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo:cu130-runtime-0123456789ab
```

Manual workflow runs default to the same standalone Dynamo image configuration.
Select `dynamo`, `vllm`, `sglang`, or `all` to override the configured
`enabled` values for one run. The default `configured` selection builds only
the enabled Dynamo image.

## One-time repository setup

Register the build machine at **Settings > Actions > Runners > New
self-hosted runner**. Keep the default `self-hosted`, `linux`, and `x64`
labels.

The runner account needs:

- GitHub Actions Runner `v2.327.1` or newer
- Docker Engine with permission to access the Docker daemon
- Docker Buildx
- Python 3
- Python packages `jinja2` and `pyyaml`
- Network access to GitHub, the source package registries used by the
  Dockerfiles, and `registry.hd-04.alayanew.com:8443`
- Enough free disk space for the selected multi-stage builds

This machine currently has about 142 GB free in Docker's data filesystem.
Keep `keep_buildkit_state` disabled until more space is available. After
enabling it, monitor usage periodically with `docker buildx du`.

Add these GitHub Actions repository secrets:

- `REGISTRY_USERNAME`
- `REGISTRY_PASSWORD`

If the private registry uses a private certificate authority, configure that
CA in the Docker daemon on the runner. Protect `DingoRouter-base` and limit
who can modify workflows because pushed workflow code executes on the local
machine.
