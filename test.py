import requests

file = {'image': open('a.jpg', 'rb')}  # Remplace avec ton image
response = requests.post('https://eaba-105-105-1-198.ngrok-free.app/predict', files=file)
print(response.json())

