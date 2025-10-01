import json
import numpy as np
"""
coordinate.py
-------------
This module calculates 2D coordinates (x, y) for hierarchical nodes in GeoMindMap.
It converts hierarchical structure (entities + parent relations) into radial layout positions.
Used to render vi_map and l_map as radial graphs.

Functions:
- calculate(nodes): main function, assign angles recursively and return {entity: (x,y)}
- calculate_coordinates(input_path, output_dir, mode): read nodes JSON, compute coords, save new JSON
"""
# Calculate Coordinates
def calculate(nodes):
    
    # Build parent-children
    children = {}
    for node in nodes:
        parent = node['parent']
        children.setdefault(parent, []).append(node['entity'])

    # Compute subtree sizes
    def subtree_size(entity):
        size = 1
        for child in children.get(entity, []):
            size += subtree_size(child)
        return size

    # Identify roots and calculate its tree total size
    roots = children[None]
    total_size = sum(subtree_size(root) for root in roots)

    # Assign angular spans to each node of a tree
    angles = {}
    def assign_angles(entity, start, end):
        # Node angle = midpoint of its span
        angles[entity] = (start + end) / 2
        # get children list
        occ = children.get(entity, [])
        if not occ:
            return
        span = end - start
        
        cum = start # start angle for children
        for child in occ:
            sz = subtree_size(child)
            child_span = span * (sz / (subtree_size(entity) - 1))  # assign span proportional to subtree size
            assign_angles(child, cum, cum + child_span) # recursive
            cum += child_span # to next child

    # Assign spans to each tree over the full circle
    cum = 0
    for root in roots:
        sz = subtree_size(root)
        span = 2 * np.pi * (sz / total_size)
        assign_angles(root, cum, cum + span)
        cum += span

    # Convert angles to coordinates
    coords = {}
    for node in nodes:
        r = node['granularity']
        theta = angles[node['entity']]
        coords[node['entity']] = (r * np.cos(theta), r * np.sin(theta))
    
    return coords

# Calculate and save coordinates to JSON
def calculate_coordinates(input_path, output_dir, mode):
    
    if mode == "vi":
        output_path = output_dir + "vi_map_layout.json"
    elif mode == "l":
        output_path = output_dir + "l_map_layout.json"

    with open(input_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    coords = calculate(nodes)
    for node in nodes:
        x, y = coords[node['entity']]
        node['x'] = x
        node['y'] = y
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
