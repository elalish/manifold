import csv
import sys
import copy
import math

# comment this out and set do_plot to False if don't have matplotlib
import matplotlib.pyplot as plt

do_plot = True

benchmark_csv_file = "benchmark.csv"

if len(sys.argv) > 1:
    benchmark_csv_file = sys.argv[1]

class TimeSeries:
    def __init__(self, name, csv_key, label):
        self.name = name
        self.csv_key = csv_key
        self.label = label
        self.series = []
        self.average = 0.0
        self.min = 0.0
        self.max = 0.0

    def set_stats(self):
        n = len(self.series)
        self.average = sum(self.series) / n if n > 0 else 0.0
        self.min = min(self.series) if n > 0 else 0.0
        self.max = max(self.series) if n > 0 else 0.0

ts_total = TimeSeries('total', 'total (ms)', 'Total time')
ts_to_man = TimeSeries('to_manifold', 'to manifold (ms)', 'Converting MeshGL to Manifold')
ts_bool = TimeSeries('boolean', 'boolean (ms)', 'Manifold boolean')
ts_from_man = TimeSeries('from_manifold', 'from manifold (ms)', 'Converting Manifold to MeshGL')

ts_to_analyze = [ts_total, ts_to_man, ts_bool, ts_from_man]
# ts_to_plot needs to be a subset of ts_to_analyze
ts_to_plot = [ts_total]
threads_key = 'threads'

def get_csv_data(filename):
    ans = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ans.append(row)
    return ans

data = get_csv_data(benchmark_csv_file)

if not data or len(data) == 0:
    print("no data")
    sys.exit(0)

if threads_key not in data[0] or any([ts.csv_key not in data[0] for ts in ts_to_analyze]):
    print("data[0]", data[0])
    print("threads key", threads_key in data[0])
    for ts in ts_to_analyze:
        print(ts.csv_key, ts.csv_key in data[0])
    print("missing a needed column in csv file")
    sys.exit(0)

thread_values_set = set([int(row[threads_key]) for row in data])
thread_values = list(thread_values_set)
thread_values.sort()
max_threads = max(thread_values)
ts_by_threads = [copy.deepcopy(ts_to_analyze) for _ in range(max_threads + 1)]
for t in thread_values:
    for ts in ts_by_threads[t]:
        ts.series = [float(row[ts.csv_key]) \
                     for row in data if int(row[threads_key]) == t]
        ts.set_stats()


def print_stats():
    for t in thread_values:
        if len(thread_values) > 1:
            print("\nThreads=", t)
        for ts in ts_by_threads[t]:
            print(ts.label)
            print("  avg",  ":", "%.3f" % ts.average)
            print("  min",  ":", "%.3f" % ts.min)
            print("  max",  ":", "%.3f" % ts.max)


def plot_hists_by_ts():
    nthreads = len(thread_values)
    nts = len(ts_to_plot)
    fig, axs = plt.subplots(nrows=nts, sharex = True)
    # plt.xscale('log')
    fig.set_figheight(6.0)
    fig.set_figwidth(8.0)
    for ts_index in range(nts):
        ts_name = ts_to_plot[ts_index].name
        ts_label = ts_to_plot[ts_index].label
        ax = axs[ts_index] if nts > 1 else axs
        all_data = []
        for threads in thread_values:
            tss = ts_by_threads[threads]
            ts = next((x for x in tss if x.name == ts_name), None)
            if not ts:
                continue
            all_data.append(ts.series)
        ax.violinplot(all_data, showmeans=True, showmedians = False,
                      showextrema = False, orientation = 'horizontal')
        ax.set_title(ts_label)
        ax.set_yticks([i+1 for i in range(nthreads)],
                      [str(k) for k in thread_values])
        ax.set_ylim(0.25, nthreads + 0.75)
        ax.set_xlim(0.0, 300.0)
        ax.set_ylabel("threads")
        ax.set_xlabel("ms")
    plt.show()

print_stats()

if do_plot:
    plot_hists_by_ts()
