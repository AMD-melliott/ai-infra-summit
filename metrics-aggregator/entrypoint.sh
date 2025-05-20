#!/bin/bash
set -e

echo "Starting metrics aggregator service..."
exec python3 metrics_aggregator.py 