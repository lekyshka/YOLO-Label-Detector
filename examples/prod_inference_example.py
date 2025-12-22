import json

import numpy as np
import requests


def main():
    payload = {"inputs": np.zeros((1, 3, 640, 640), dtype="float32").tolist()}
    r = requests.post(
        "http://127.0.0.1:5001/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    return r.json()


if __name__ == "__main__":
    print(main())
