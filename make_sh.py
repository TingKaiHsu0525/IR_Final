import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Generate a shell script that runs a Python file with args from a JSON file."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the JSON file containing the arguments."
    )
    parser.add_argument(
        "python_file",
        type=str,
        help="The Python script (e.g. app.py) you want to invoke."
    )
    args = parser.parse_args()

    # 1. Load the JSON file
    try:
        with open(args.json_path, "r") as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error: could not load JSON from {args.json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Build the list of lines for the shell script
    lines = []
    # First line: "python <python_file> \"
    lines.append(f"python {args.python_file} \\")
    # Then one line per key/value pair
    #   --key <value> \
    for key, value in params.items():
        # Format the value:
        if isinstance(value, str):
            # wrap string in double quotes
            val_str = f"\"{value}\""
        elif isinstance(value, bool):
            # lowercase JSON‐style booleans
            val_str = str(value).lower()
        else:
            # number (int/float) or other JSON types → convert to str()
            val_str = str(value)
        lines.append(f"    --{key} {val_str} \\")
    # Remove the trailing backslash from the last line
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(" \\")

    # 4. Write out the shell script
    sh_fn = args.python_file + ".sh"
    try:
        with open(sh_fn, "w") as out:
            out.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"Error: could not write to {sh_fn}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Shell script written to: {sh_fn}")

if __name__ == "__main__":
    main()
