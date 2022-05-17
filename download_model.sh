#!/bin/bash

file_id=$1
echo "[id: $file_id]"

gdown $file_id -O DETR.model