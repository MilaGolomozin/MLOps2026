from locust import HttpUser, between, task


class ApiUser(HttpUser):
    """Simulate a user calling the API."""

    wait_time = between(1, 2)

    @task(3)
    def read_root(self) -> None:
        """Request the root endpoint."""
        self.client.get("/")

    @task(1)
    def generate_image(self) -> None:
        """Request the generate endpoint."""
        self.client.post("/generate", json={"batch_size": 1, "n_sample_steps": 1})
