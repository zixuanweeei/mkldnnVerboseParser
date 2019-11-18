# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MKLDNN profiler parser.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--file", "-f", default="",
                      help="MKLDNN profiler file")
  parser.add_argument("--run", "-r", default=200, type=int,
                      help="RNN iterations.")
  args = parser.parse_args()

  verbose = pd.read_csv(args.file, header=None, delimiter=',')
  verbose.columns = ['Item', 'Status', 'Device', 'Op', 'Type', 'Prop', 'Format', 'Num', 'Shape',
                     'Millisec']
  verbose.fillna('gemm', inplace=True)

  # check repeat pattern
  last_verbose = verbose.iloc[-1, :]
  row_id = verbose.shape[0] - 2
  for _ in range(verbose.shape[0]):
    equality = last_verbose.loc['Op':'Shape'] == verbose.loc[row_id, 'Op':'Shape']
    if equality.all():
      span = verbose.shape[0] - 1 - row_id
      row = verbose.shape[0] - 1
      exact = True
      for i in range(span):
        eqly = verbose.loc[row - i, 'Op':'Shape'] == verbose.loc[row - i - span, 'Op':'Shape']
        if not eqly.all():
          exact = False
      if exact:
        break
    row_id -= 1

  dryrun = args.run
  drystart = verbose.shape[0] - dryrun * span
  dryverbose = verbose.loc[drystart:, :]
  print("Span: {}, Log number: {}".format(span, dryverbose.shape[0]))
  shapes = dryverbose.loc[:, 'Shape'].values.reshape((-1, span))
  if not (shapes == shapes[0]).all():
    raise RuntimeError("Dirty verbose.")
  millisecs = dryverbose.loc[:, 'Millisec'].values.reshape((-1, span)).transpose()
  millisecs = pd.DataFrame(millisecs, columns=[str(i) for i in range(millisecs.shape[-1])])
  millisecs_aggregate = millisecs.agg(['sum', 'mean', 'min', 'max'], axis=1).reset_index(drop=True)
  items = dryverbose.iloc[:span, :-1].reset_index(drop=True)
  result = pd.concat([items, millisecs_aggregate], axis=1)
  filename = os.path.splitext(args.file)[0] + ".xlsx"
  print(filename)
  result.to_excel(filename, index=False)

  ops = result.loc[:, ['Op', 'sum']]
  ops_aggregate = ops.groupby('Op').agg(['count', 'sum'])
  print(ops_aggregate)
