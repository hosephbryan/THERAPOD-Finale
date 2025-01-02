from transformers import pipeline
import requests

def check_suicidality(text):
    classifier = pipeline("sentiment-analysis", model="sentinetyd/suicidality", device_map="cpu")
    result = classifier(text)

    if result[0]['label'] == 'LABEL_1':
        return True
    
    return False

def critical_notif(user_input):
    is_critical = check_suicidality(user_input)

    if is_critical:
        url = "https://www.falcore-therapod.com/create_notification/"
    
        payload = {
            "client_id": "1",
            "doctor_id": "2",
            "message": user_input
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 201:
            print("Notification created successfully.")
        else:
            print(f"Failed to create notification. Status Code: {response.status_code}")
