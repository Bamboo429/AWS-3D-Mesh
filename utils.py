import os
#import yaml
from typing import Dict, Any


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)

'''
def load_yaml(x: str) -> Dict[str, Any]:
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        return config
'''    
    
def calculate_distance(marker_size, focal_length, apparent_size):
    # Distance (depth) = (Marker size * Focal length) / Apparent size
    distance = (marker_size * focal_length) / apparent_size
    return distance
