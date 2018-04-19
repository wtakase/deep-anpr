#!/usr/bin/env python

import os
import sys


def main():
    in_file_name = sys.argv[1]
    file_name_base, _ = os.path.splitext(in_file_name)
    out_file_name = "%s.csv" % file_name_base
    with open(in_file_name) as in_f:
        records = []
        for line in in_f:
            record = {}
            items = line.strip().split(",")
            for item in items:
                key = item.split(":")[0].strip()
                value = item.split(":")[1].strip().replace("%", "")
                record[key] = value
            records.append(record)
        records.pop()
        with open(out_file_name, "w") as out_f:
            out_f.write("# %s\n" % ", ".join(records[0].keys()))
            for record in records:
                out_f.write("%s\n" % ", ".join(record.values()))


if __name__ == "__main__":
    main()
