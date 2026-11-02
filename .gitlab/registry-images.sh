#!/bin/sh
# Both modes are read-only at XB01; only sync writes the selected destination.
set +x
set -eu
[ "${CI_DEBUG_TRACE:-false}" = false ] || { printf 'Disable CI_DEBUG_TRACE.\n' >&2; exit 1; }
: "${CI_IMAGE_REGISTRY_USER_XB01:?Set CI_IMAGE_REGISTRY_USER_XB01}"
: "${CI_IMAGE_REGISTRY_PASSWORD_XB01:?Set CI_IMAGE_REGISTRY_PASSWORD_XB01}"
umask 077
auth_dir="$(mktemp -d /tmp/dingo-sync-auth.XXXXXX)"
trap 'rm -rf "$auth_dir"' 0
trap 'exit 1' HUP INT TERM
export REGISTRY_AUTH_FILE="$auth_dir/auth.json"
printf '%s' "$CI_IMAGE_REGISTRY_PASSWORD_XB01" | skopeo login \
    --username "$CI_IMAGE_REGISTRY_USER_XB01" --password-stdin "$CI_IMAGE_REGISTRY_HOST"
unset CI_IMAGE_REGISTRY_PASSWORD_XB01
if [ "${1:-}" = check-bases ]; then
    while IFS="$(printf '\t')" read -r source target; do
        printf 'Required mirror: %s -> %s\n' "$source" "$target"
        inspect_ref="$target"
        # Skopeo rejects tag@digest although Docker/BuildKit accept it.
        case "$target" in
            *@*)
                repository="${target%@*}"
                case "${repository##*/}" in *:*) repository="${repository%:*}" ;; esac
                inspect_ref="${repository}@${target##*@}"
                ;;
        esac
        skopeo inspect --raw --retry-times 3 "docker://$inspect_ref" > /dev/null || {
            printf 'Missing/inaccessible XB01 base. Copy it using sync-router-public-to-xb01.sh first.\n' >&2
            exit 1
        }
    done < .gitlab-render/bases.tsv
    exit 0
fi
case "${CI_SYNC_REGION:-}" in
    HD04)
        target_host=registry.hd-04.alayanew.com:8443
        target_user="${CI_IMAGE_REGISTRY_USER_HD04:?Set CI_IMAGE_REGISTRY_USER_HD04}"
        target_password="${CI_IMAGE_REGISTRY_PASSWORD_HD04:?Set CI_IMAGE_REGISTRY_PASSWORD_HD04}"
        ;;
    BJ09)
        target_host=registry.bj-09.alayanew.com:8443
        target_user="${CI_IMAGE_REGISTRY_USER_BJ09:?Set CI_IMAGE_REGISTRY_USER_BJ09}"
        target_password="${CI_IMAGE_REGISTRY_PASSWORD_BJ09:?Set CI_IMAGE_REGISTRY_PASSWORD_BJ09}"
        ;;
    *) printf 'Unsupported sync region\n' >&2; exit 1 ;;
esac
printf '%s' "$target_password" | skopeo login --username "$target_user" --password-stdin "$target_host"
unset target_password
while IFS="$(printf '\t')" read -r image digest; do
    [ -n "$image" ] || continue
    case "$image" in "$CI_IMAGE_REGISTRY_HOST"/openclaw/*:*) ;; *) exit 1 ;; esac
    printf '%s\n' "$digest" | grep -Eq '^sha256:[a-f0-9]{64}$' || exit 1
    source_ref="${image%:*}@${digest}"
    target_image="${target_host}/${image#*/}"
    skopeo copy --all --preserve-digests --retry-times 3 "docker://$source_ref" "docker://$target_image"
    actual="$(skopeo inspect --retry-times 3 --format '{{.Digest}}' "docker://$target_image")"
    [ "$digest" = "$actual" ] || { printf 'Digest mismatch for %s\n' "$target_image" >&2; exit 1; }
    printf 'Synced %s (%s)\n' "$target_image" "$actual"
done < .gitlab-published/images.tsv
