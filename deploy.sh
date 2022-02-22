#!/bin/bash

##
# Copyright 2020 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

set -e
set -x

DEPLOYABLE_STATIC_FILES=( \
  samples/models \
)

PACKAGE_ROOT=`pwd`
DEPLOY_ROOT=$PACKAGE_ROOT/docs/html

function copyToDeployRoot {
  path=$1

  echo "Copying $path"

  if [ -d "$path" ]; then
    directory="$path"
  else
    directory="`dirname $path`"
  fi

  echo "Creating $DEPLOY_ROOT/$directory"
  mkdir -p "$DEPLOY_ROOT/$directory"

  if [ -d "${path}" ]; then
    cp -r $path/* "$DEPLOY_ROOT/$path"
  else
    if [ -f "${path}" ]; then
      cp $path "$DEPLOY_ROOT/$path"
    else
      echo "Path not found: $path"
      exit 1
    fi
  fi
}

mkdir -p $DEPLOY_ROOT
touch $DEPLOY_ROOT/.nojekyll

# Copy over deployable static files and directories, maintaining relative paths
for static in "${DEPLOYABLE_STATIC_FILES[@]}"; do
  echo $static
  copyToDeployRoot $static
done

set -x

# Add a "VERSION" file containing the last git commit message
git log -n 1 > $DEPLOY_ROOT/VERSION

git status --ignored

set +e
set +x