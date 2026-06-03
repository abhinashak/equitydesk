#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests


SECRET_FILE = ".secret/kite.secret"


def load_secrets(path):
    secrets = {}

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()

    return secrets


def get_request_token(login_url):
    qs = parse_qs(urlparse(login_url).query)

    status = qs.get("status", [""])[0]

    if status != "success":
        raise Exception(f"Login failed. status={status}")

    return qs["request_token"][0]


def main():
    login_url = input("Paste Zerodha login URL: ").strip()

    secrets = load_secrets(SECRET_FILE)

    api_key = secrets["api_key"]
    api_secret = secrets["api_secret"]

    request_token = get_request_token(login_url)

    payload = api_key + request_token + api_secret

    checksum = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    response = requests.post(
        "https://api.kite.trade/session/token",
        headers={
            "X-Kite-Version": "3"
        },
        data={
            "api_key": api_key,
            "request_token": request_token,
            "checksum": checksum
        }
    )

    response.raise_for_status()

    result = response.json()

    if result["status"] != "success":
        raise Exception(json.dumps(result, indent=2))

    access_token = result["data"]["access_token"]

    print()
    print(f"{api_key}:{access_token}")


if __name__ == "__main__":
    main()
