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

#include "test.h"

Options options;

void print_usage() {
  printf("-------------------------------\n");
  printf("manifold_test specific options:\n");
  printf("  -h: Print this message\n");
  printf("  -e: Export sample models\n");
  printf("  -v: Enable verbose output\n");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int c;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-') {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage();
      return 1;
    }
    switch (argv[i][1]) {
      case 'h':
        print_usage();
        return 0;
      case 'e':
        options.exportModels = true;
        break;
      case 'v':
        options.params.verbose = true;
        break;
      default:
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage();
        return 1;
    }
  }

  return RUN_ALL_TESTS();
}
