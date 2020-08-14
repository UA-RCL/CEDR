import argparse
import sys
import re

import lief

def instrument_bin(bin_file, dag_file, out_file=None):
    if out_file == None:
        out_file = f"{bin_file}.inst"

    with open(dag_file) as dag_handle:
        dag_str = dag_handle.read()
        dag_str = re.sub(r"[ \n\r\t]", "", dag_str) + "\0"
        dag_bytes = list(map(ord, dag_str))

        input_bin = lief.parse(bin_file)
        if not input_bin:
            print(f"Failed to parse input binary {bin_file}", file=sys.stderr)
            return 1

        input_bin.add(lief.ELF.Note("DASH", lief.ELF.NOTE_TYPES.UNKNOWN, dag_bytes))
        input_bin.write(out_file)
    
    return 0


def generate_argparser():
    parser = argparse.ArgumentParser(description="Use LIEF to instrument a given ELF binary with a JSON DAG-file")
    parser.add_argument("-d", "--dag",
                        help="File containing JSON-based input DAG to be included with your binary", 
                        metavar="DAG_FILE", type=str, required=True, dest="dag_file")
    parser.add_argument("-b", "--binary",
                        help="The input binary to instrument",
                        metavar="BIN_FILE", type=str, required=True, dest="bin_file")
    parser.add_argument("-o", "--output",
                        help="The output filename. Default is BIN_FILE.inst to indicate that the binary has been instrumented",
                        metavar="OUT_FILE", type=str, required=False, dest="out_file", default=None)
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()
    sys.exit(instrument_bin(args.bin_file, args.dag_file, args.out_file))