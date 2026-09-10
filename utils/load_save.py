# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import yaml
from typing import Dict, Any

def write_data(data: Dict[str, Any], filename: str) -> None:
    """
    Writes a dictionary to a file in JSON, YAML, or PKL format based on the file extension.
    
    Parameters:
        data (dict): The dictionary to write.
        filename (str): The output filename with extension (.json, .yaml, .yml, or .pkl).
    """
    extension = filename.split('.')[-1].lower()

    try:
        if extension == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        elif extension in {'yaml', 'yml'}:
            with open(filename, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif extension == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file format. Use '.json', '.yaml', '.yml', or '.pkl'.")
    except Exception as e:
        print(f"Error writing file: {e}")


def load_data(filename: str) -> Dict[str, Any]:
    """
    Loads data from a JSON, YAML, or PKL file and returns it as a dictionary.
    
    Parameters:
        filename (str): The input filename with extension (.json, .yaml, .yml, or .pkl).
        
    Returns:
        dict: The loaded data as a dictionary.
    """
    extension = filename.split('.')[-1].lower()

    try:
        if extension == 'json':
            with open(filename, 'r') as f:
                return json.load(f)
        elif extension in {'yaml', 'yml'}:
            with open(filename, 'r') as f:
                return yaml.safe_load(f)
        elif extension == 'pkl':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use '.json', '.yaml', '.yml', or '.pkl'.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}