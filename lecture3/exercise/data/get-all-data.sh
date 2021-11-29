#! /bin/bash

# For each file, add a download.py line
# Any additional processing on the downloaded file

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Yelp Reviews dataset
mkdir -p surnames
if [ ! -f surnames/surnames.csv ]; then
    python download.py 1MBiOU5UCaGpJw2keXAqOLL8PCJg_uZaU surnames/surnames.csv # 6
fi
if [ ! -f surnames/surnames_with_splits.csv ]; then
    python download.py 1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU- surnames/surnames_with_splits.csv # 8
fi
