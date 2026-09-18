#!/usr/bin/env sh
set -eu
base_url="${1:?base URL required}"
curl --fail --silent --show-error "${base_url}/health/ready"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"pickup_datetime":"2025-01-15T08:30:00-05:00","pickup_location_id":132,"dropoff_location_id":230,"passenger_count":1}' \
  "${base_url}/v1/predict"

