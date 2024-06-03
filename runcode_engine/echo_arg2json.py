import json
import sys

def transform_to_json(args):
    # Assuming args are in key-value pairs, e.g., "--name John --age 25"
    result = {}
    current_key = None

    for arg in args[1:]:  # Skipping the script name (args[0])
        if arg.startswith('--'):
            current_key = arg[2:]
            result[current_key] = None
        elif arg.startswith('-'):
            current_key = arg[1:]
            result[current_key] = None
        elif current_key is not None:
            # if arg is float
            arg: str = arg
            if arg.isdigit():
                result[current_key] = int(arg)
            elif arg.lower() == "true":
                result[current_key] = True
            elif arg.lower() == "false":
                result[current_key] = False
            else:
                try:
                    result[current_key] = float(arg)
                except ValueError:
                    result[current_key] = arg
            current_key = None
    return json.dumps(result)

if __name__ == "__main__":
    json_output = transform_to_json(sys.argv)
    print(json_output)
