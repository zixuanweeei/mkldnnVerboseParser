#! /usr/bin/python
# coding: utf-8
import io
import argparse

import pandas as pd


def parse_all(string):
  if string == "":
    print("Empty input. Exits.")
    return
  mode = "exec" if args.exec else "create"

  buffer = io.StringIO(string)
  lines = []
  iterno = 0
  for line in buffer.readlines():
    if "mkldnn_verbose" in line and mode in line:
      lines.append(str(iterno) + ", " + line)
    if "====" in line or "exec,rnn," in line:
      iterno += 1
  csv_string = "\n".join(lines)
  csv_buffer = io.StringIO(csv_string)
  csv_data = pd.read_csv(csv_buffer, header=None)
  csv_data.columns = ["iter", "flag1", "flag2", "op", "flag3", "flag4",
      "desc", "num", "dim", "cost"]
  csv_data.drop(columns=["flag1", "flag2", "flag3", "flag4"], inplace=True)
  stat = csv_data.groupby(["iter", "op", "desc", "num", "dim"])["cost"].sum()
  # stat.drop(columns=["cost"], inplace=True)
  stats = stat.unstack(level=0)
  # iter 0
  concat0 = stats.loc["concat", 0].sum()
  reorder0 = stats.loc["reorder", 0].sum()
  rnn0 = stats.loc["rnn", 0].sum()

  # iter warm-up
  if args.warm > 1:
    concat_w = stats.loc["concat", 1:args.warm - 1].sum(axis=0).mean()
    reorder_w = stats.loc["reorder", 1:args.warm - 1].sum(axis=0).mean()
    rnn_w = stats.loc["rnn", 1:args.warm - 1].sum(axis=0).mean()

  # iter 1 ~
  concat1 = stats.loc["concat", args.warm - 1 if args.warm > 1 else 1:].sum(axis=0).mean()
  reorder1 = stats.loc["reorder", args.warm - 1 if args.warm > 1 else 1:].sum(axis=0).mean()
  rnn1 = stats.loc["rnn", args.warm - 1 if args.warm > 1 else 1:].sum(axis=0).mean()
  if args.warm <= 1:
    result = pd.DataFrame({"iter0 (ms)": [concat0, reorder0, rnn0],
                          "iter1~" + str(args.iter - 1) + " (ms)": [concat1, reorder1, rnn1]},
                          index=["Concat", "Reorder", "RNN"])
  else:
    result = pd.DataFrame({"iter0 (ms)": [concat0, reorder0, rnn0],
                          "iter1~" + str(args.warm - 1) + " (ms)": [concat_w, reorder_w, rnn_w],
                          "iter{}~{}".format(args.warm, args.iter - 1) + " (ms)": [concat1, reorder1, rnn1]},
                          index=["Concat", "Reorder", "RNN"])
  total = result.sum(axis=0)
  print("\n")
  print(result)
  print(total)
  print("\n")

  if args.dest != "":
    result.to_csv(args.dest, index=False)

  # writer = pd.ExcelWriter("MKLDNN_profiler.xlsx", mode="a")
  # stats.to_excel(writer, sheet_name=mode)
  # writer.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MKLDNN profiler parser.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--file", "-f", default="",
                      help="MKLDNN profiler file")
  parser.add_argument("--str", "-s", default="",
                      help="MKLDNN output string.")
  parser.add_argument("--iter", "-i", default=10, type=int,
                      help="RNN iterations.")
  parser.add_argument("--warm", "-w", type=int, default=1,
                      help="Number of warm-up loop.")
  parser.add_argument("--dest", "-d", type=str, default="",
                      help="Dest file to store the result.")
  parser.add_argument("--create", dest="exec", action="store_false")
  parser.set_defaults(exec=True)
  args = parser.parse_args()

  if args.file == "":
    if args.str == "":
      args.str = input("Please input the MKLDNN outputs: ")
    parse_all(args.str)
  else:
    verbose = open(args.file, "r+")
    string = verbose.read()
    parse_all(string)
    verbose.close()
