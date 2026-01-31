#!/bin/bash

# Parameter check
if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi

# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
RAY_PORT=${RAY_PORT:-29500}  # Default port for Ray, can be modified if needed

# Head node startup logic
if [ "$RANK" -eq 0 ]; then
    # Start Ray head node and capture output
    echo "Starting Ray head node on rank 0"
    RAY_START_OUTPUT=$(ray start --head --memory=461708984320 --port=$RAY_PORT 2>&1)
    
    # Extract the head node address from ray start output
    # The output contains: ray start --address='IP:PORT'
    HEAD_ADDRESS_FULL=$(echo "$RAY_START_OUTPUT" | grep "ray start --address=" | sed -n "s/.*ray start --address='\([^']*\)'.*/\1/p" | head -n 1)
    
    if [ -n "$HEAD_ADDRESS_FULL" ]; then
        # Use the full address (IP:PORT) extracted from ray start output
        HEAD_ADDRESS=$HEAD_ADDRESS_FULL
        echo "Head node address extracted from ray start output: $HEAD_ADDRESS"
    else
        # Fallback: use ray list node to find the alive head node IP
        echo "Could not extract address from ray start output, trying 'ray list node'..."
        sleep 2
        HEAD_IP=""
        for i in {1..30}; do
            HEAD_IP=$(ray list node --address="localhost:$RAY_PORT" 2>/dev/null | grep -i "alive" | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -n 1)
            if [ -n "$HEAD_IP" ]; then
                break
            fi
            sleep 1
        done
        
        # Final fallback to hostname
        if [ -z "$HEAD_IP" ]; then
            echo "Warning: Could not detect head IP via 'ray list node', falling back to hostname"
            HEAD_IP=$(hostname -I | awk '{print $1}')
        fi
        
        # Construct full address with port
        HEAD_ADDRESS="$HEAD_IP:$RAY_PORT"
    fi
    
    echo "Head node detected at address: $HEAD_ADDRESS"
    # Write full address (IP:PORT) to file
    echo "$HEAD_ADDRESS" > $RAY_HEAD_IP_FILE
    echo "Head node address written to $RAY_HEAD_IP_FILE"
else
    # Worker node startup logic
    echo "Waiting for head node IP file..."
    
    # Wait for file to appear (wait up to 360 seconds)
    for i in {1..360}; do
        if [ -f $RAY_HEAD_IP_FILE ]; then
            HEAD_ADDRESS=$(cat $RAY_HEAD_IP_FILE)
            if [ -n "$HEAD_ADDRESS" ]; then
                break
            fi
        fi
        sleep 1
    done
    
    if [ -z "$HEAD_ADDRESS" ]; then
        echo "Error: Could not get head node address from $RAY_HEAD_IP_FILE"
        exit 1
    fi
    
    echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
    ray start --memory=461708984320 --address="$HEAD_ADDRESS"
fi