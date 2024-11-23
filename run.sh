#!/bin/bash

# Define environment variables
./setenv.sh

# Define the directory containing the Python files
DIRECTORY="style-transfer-modules"

# Define the JSON file specifying file-env pairs
JSON_FILE="module_env_map.json"

# Verify the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist."
    exit 1
fi

# Change to the specified directory
cd "$DIRECTORY" || exit 1

# Verify the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file '$JSON_FILE' does not exist."
    exit 1
fi

# Enable Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Use Python to parse the JSON and get file-env pairs
python_files_and_envs=$(python3 <<EOF
import json
with open("$JSON_FILE") as f:
    data = json.load(f)
for file, env in data.items():
    print(f"{file} {env}")
EOF
)

# Process each Python file and its corresponding Conda environment
while IFS= read -r line; do
    # Extract file name and environment name
    FILE_NAME=$(echo "$line" | awk '{print $1}')
    ENV_NAME=$(echo "$line" | awk '{print $2}')

    # Check if the Python file exists
    if [ ! -f "$FILE_NAME" ]; then
        echo "Error: Python file '$FILE_NAME' not found in directory '$DIRECTORY'. Skipping."
        continue
    fi

    # Activate the specified Conda environment
    echo "Activating Conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"

    # Verify the environment was activated successfully
    if [ $? -ne 0 ]; then
        echo "Error: Conda environment '$ENV_NAME' could not be activated. Skipping $FILE_NAME."
        continue
    fi

    # Run the Python file
    echo "Running $FILE_NAME in environment $ENV_NAME..."
    python "$FILE_NAME"

    # Check for errors during execution
    if [ $? -ne 0 ]; then
        echo "Error: $FILE_NAME encountered an issue during execution."
        # Optionally exit or continue with the next file
        continue
    fi

    # Deactivate the Conda environment
    conda deactivate
done <<< "$python_files_and_envs"

echo "Script execution completed."
