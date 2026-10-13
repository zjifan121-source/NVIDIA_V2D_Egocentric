#!/usr/bin/env python3
"""Resolve Ryan's public GHCR images without Docker or registry credentials."""
import argparse
import concurrent.futures
import datetime
import json
import urllib.request
from pathlib import Path

MODULES = 'anycalib moge grounding_dino sam2 sam3d foundation_pose wilor hamer geocalib droid_slam gsplat_refinement mediapipe'.split()
ACCEPT = ', '.join(['application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json', 'application/vnd.oci.image.manifest.v1+json', 'application/vnd.docker.distribution.manifest.v2+json'])


def resolve(module, tag):
    repo = 'ryanzhou123232/v2d_' + module
    with urllib.request.urlopen('https://ghcr.io/token?scope=repository:' + repo + ':pull', timeout=60) as response:
        token = json.load(response)['token']

    def manifest(ref):
        request = urllib.request.Request('https://ghcr.io/v2/' + repo + '/manifests/' + ref,
                                         headers={'Authorization': 'Bearer ' + token, 'Accept': ACCEPT})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response), response.headers['Docker-Content-Digest']

    index, digest = manifest(tag)
    if 'manifests' in index:
        candidates = [m for m in index['manifests'] if m.get('platform', {}).get('os') == 'linux' and m['platform'].get('architecture') == 'amd64']
        assert len(candidates) == 1, (repo, candidates)
        image, platform_digest = manifest(candidates[0]['digest'])
    else:
        image, platform_digest = index, digest
    return {'image': 'ghcr.io/' + repo, 'tag': tag, 'digest': digest,
            'platform_digest': platform_digest, 'local_tag': 'v2d_' + module + ':latest',
            'compressed_bytes': sum(layer['size'] for layer in image['layers']),
            'layers': [{'digest': layer['digest'], 'size': layer['size']} for layer in image['layers']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='afe7dad')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(lambda module: resolve(module, args.tag), MODULES))
    unique = {layer['digest']: layer['size'] for image in images for layer in image['layers']}
    record = {'resolved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'platform': 'linux/amd64', 'unique_compressed_bytes': sum(unique.values()), 'images': images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    for image in images:
        print(image['local_tag'], f"{image['compressed_bytes']/2**30:.2f} GiB compressed", image['digest'])
    print(f"Unique compressed layers: {sum(unique.values())/2**30:.2f} GiB. Extracted storage is larger.")


if __name__ == '__main__':
    main()
