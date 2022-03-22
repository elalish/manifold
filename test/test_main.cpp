// Copyright 2022 Emmett Lalish
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>

#include "test.h"

Options options;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  char *cvalue = NULL;
  int c;
  opterr = 0;

  while ((c = getopt(argc, argv, "ev")) != -1) switch (c) {
      case 'e':
        options.exportModels = true;
        break;
      case 'v':
        options.params.verbose = true;
        break;
        //   case 'c':
        //     cvalue = optarg;
        //     break;
      case '?':
        // if (optopt == 'c')
        //   fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default:
        abort();
    }

  return RUN_ALL_TESTS();
}