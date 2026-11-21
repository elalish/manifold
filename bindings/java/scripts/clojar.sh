missing_vars=()
[[ -z "${CLOJARS_USERNAME}" ]] && missing_vars+=("CLOJARS_USERNAME")
[[ -z "${CLOJARS_TOKEN}" ]] && missing_vars+=("CLOJARS_TOKEN")
[[ -z "${JAVA_HOME}" ]] && missing_vars+=("JAVA_HOME")

if [[ ${#missing_vars[@]} -gt 0 ]]; then
  echo "Error: The following required environment variables are not set: ${missing_vars[*]}"
  echo "Create them at https://clojars.org/profile"
  exit 1
fi

echo "Clojar Java $JAVA_HOME"
echo "Clojar Version: $1"

set -e
cd bindings/java/
mvn versions:set -DnewVersion=$1 --file pom.xml  --no-transfer-progress
mvn deploy --settings=settings.xml --no-transfer-progress
cd ../../
set -e
