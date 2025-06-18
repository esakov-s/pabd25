from locust import FastHttpUser, task, between
import dotenv

config = dotenv.dotenv_values(".env")
token = config["APP_TOKEN"]
data = {"area": 44, "rooms": 2, "total_floors": 5, "floor": 3}

class WebsiteTestUser(FastHttpUser):
    wait_time = between(0.5, 3.0)

    @task
    def test_api(self):
        with self.rest("POST", "/api/numbers", 
                       json=data, 
                       headers={
                           "Authorization": f"Bearer {token}", 
                           "Content-Type": "application/json"
                           }) as resp:
            if resp.js["status"] != "success":
                resp.failure(f"Unexpected value in response {resp.text}")
           