missing_vars=()
[[ -z "${CLOJARS_USERNAME}" ]] && missing_vars+=("CLOJARS_USERNAME")
[[ -z "${CLOJARS_TOKEN}" ]] && missing_vars+=("CLOJARS_TOKEN")
[[ -z "${JAVA_HOME}" ]] && missing_vars+=("JAVA_HOME")

if [[ ${#missing_vars[@]} -gt 0 ]]; then
  echo "Error: The following required environment variables are not set: ${missing_vars[*]}"
  echo "Create them at https://clojars.org/profile"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=$(bash $SCRIPT_DIR/findVersion.sh)
echo "Clojar Java $JAVA_HOME"
echo "Clojar Version: $VERSION"

set -e
cd bindings/java/
mvn versions:set -DnewVersion=$VERSION --file pom.xml  --no-transfer-progress
mvn install
cd ../../
set -e
