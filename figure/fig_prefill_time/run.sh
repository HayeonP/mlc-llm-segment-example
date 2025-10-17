#!/bin/bash

MAX_TOKENS=256

./profile_prefill_time $MAX_TOKENS 16 > output_chunk_16.txt
./profile_prefill_time $MAX_TOKENS 32 > output_chunk_32.txt
./profile_prefill_time $MAX_TOKENS 64 > output_chunk_64.txt
./profile_prefill_time $MAX_TOKENS 128 > output_chunk_128.txt
./profile_prefill_time $MAX_TOKENS 256 > output_chunk_256.txt