import requests

def send_post_request(client_id, summary):
    url = "https://www.falcore-therapod.com/api/sessions/"  # Replace with your endpoint URL
    headers = {
        "Authorization": "therapox",
        "Content-Type": "application/json"
    }
    payload = {
        "client": client_id,
        "summary": summary
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            print("Request successful!", response.json())
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
