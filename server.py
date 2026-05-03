# FILE: server.py
# PURPOSE: Digital Twin Hub with ROBUST SMART MERGING

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure Logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

app = Flask(__name__)
CORS(app)

# The "Database"
digital_twins = {}

print("------------------------------------------------")
print("   DIGITAL TWIN HUB (SAFE MERGE ACTIVE)         ")
print("   Status: ONLINE on Port 5000                  ")
print("------------------------------------------------")

def deep_merge(source, destination):
    """
    Safely merges source into destination.
    Creates dictionaries if they are missing/None.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # Get the node from destination
            node = destination.get(key)
            
            # If node is missing or None (not a dict), create an empty one
            if node is None or not isinstance(node, dict):
                destination[key] = {}
                node = destination[key]
            
            # Recursive call
            deep_merge(value, node)
        else:
            # Simple value (update directly)
            destination[key] = value
    return destination

@app.route('/api/2/things/<thing_id>', methods=['PUT'])
def update_thing(thing_id):
    new_data = request.json
    
    if thing_id not in digital_twins:
        # First time seeing this twin? Create it.
        digital_twins[thing_id] = new_data
    else:
        # Twin exists? SAFELY MERGE
        deep_merge(new_data, digital_twins[thing_id])
    
    return jsonify(digital_twins[thing_id]), 201

@app.route('/api/2/things/<thing_id>', methods=['GET'])
def get_thing(thing_id):
    return jsonify(digital_twins.get(thing_id, {})), 200

@app.route('/api/2/things', methods=['GET'])
def get_all():
    return jsonify(list(digital_twins.values())), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)