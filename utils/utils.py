import json

def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return its content as a dictionary.
    Args:
        - file_path (str): The path to the JSON file.
    Returns:
    - dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data