git fetch --tags
TAG=$(git tag --sort=-creatordate | head -1)
if [ -z "$TAG" ]; then
  TAG="0.0.0"
fi
EXACT=$(git describe --tags --exact-match 2>/dev/null || true)
if [ -z "$EXACT" ]; then
  HASH=$(git rev-parse --short HEAD)
  export VERSION="${TAG}-${HASH}-SNAPSHOT"
else
  export VERSION="${TAG}"
fi
echo "$VERSION"
