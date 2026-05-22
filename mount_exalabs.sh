#!/usr/bin/env bash
set -euo pipefail

BUCKET="exalabs"
MOUNT_DIR="/s3"
AWS_PROFILE_NAME="exalabs"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Using AWS profile: ${AWS_PROFILE_NAME}"
echo "Bucket: s3://${BUCKET}"
echo "Mount point: ${MOUNT_DIR}"
echo

# Install AWS CLI if missing
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install it first:"
  echo "  sudo apt update && sudo apt install -y awscli"
  exit 1
fi

# Install Mountpoint for S3 if missing
if ! command -v mount-s3 >/dev/null 2>&1; then
  echo "mount-s3 not found. Installing Mountpoint for Amazon S3..."

  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) MP_URL="https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb" ;;
    aarch64|arm64) MP_URL="https://s3.amazonaws.com/mountpoint-s3-release/latest/arm64/mount-s3.deb" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
  esac

  TMP_DEB="$(mktemp --suffix=.deb)"
  wget -O "$TMP_DEB" "$MP_URL"
  sudo apt-get install -y "$TMP_DEB"
  rm -f "$TMP_DEB"
fi

echo
read -r -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -r -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo

mkdir -p ~/.aws
chmod 700 ~/.aws

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" --profile "$AWS_PROFILE_NAME"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile "$AWS_PROFILE_NAME"
aws configure set region "$AWS_REGION" --profile "$AWS_PROFILE_NAME"
aws configure set output json --profile "$AWS_PROFILE_NAME"

chmod 600 ~/.aws/credentials ~/.aws/config 2>/dev/null || true

echo
echo "Testing access to s3://${BUCKET}..."
aws s3 ls "s3://${BUCKET}" --profile "$AWS_PROFILE_NAME" >/dev/null

echo "Preparing mount directory..."
sudo mkdir -p "$MOUNT_DIR"
sudo chown "$USER:$USER" "$MOUNT_DIR"

if mountpoint -q "$MOUNT_DIR"; then
  echo "$MOUNT_DIR is already mounted."
  exit 0
fi

echo "Mounting s3://${BUCKET} to ${MOUNT_DIR}..."
sudo mount-s3 --profile "$AWS_PROFILE_NAME" "s3://${BUCKET}/" "$MOUNT_DIR"

echo
echo "Mounted successfully:"
df -h "$MOUNT_DIR" || true
ls -la "$MOUNT_DIR" | head
