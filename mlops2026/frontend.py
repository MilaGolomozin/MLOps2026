"""Streamlit UI that calls the FastAPI backend generate endpoint.

The backend owns the model and weights and performs inference in
vdm_pokemon.api.generate.
"""

import os

import requests
import streamlit as st


def get_backend_url() -> str:
    """Return the backend URL for the API.

    Returns:
        The backend URL.
    """
    return os.environ.get("BACKEND", "http://127.0.0.1:8000")


def build_payload(n_sample_steps: int) -> dict[str, int]:
    """Build the JSON payload for the generate endpoint.

    Args:
        n_sample_steps: The number of sampling steps to run.

    Returns:
        The payload dictionary for the request.
    """
    return {"batch_size": 1, "n_sample_steps": n_sample_steps}


def request_image(backend_url: str, n_sample_steps: int) -> bytes:
    """Request a generated image from the backend inference endpoint.

    Args:
        backend_url: The backend base URL.
        n_sample_steps: The number of sampling steps to run.

    Returns:
        The PNG bytes returned by the backend after model inference.
    """
    url = f"{backend_url.rstrip('/')}/generate"
    response = requests.post(url, json=build_payload(n_sample_steps), timeout=120)
    response.raise_for_status()
    return response.content


def main() -> None:
    """Run the Streamlit UI."""
    st.title("VDM Pokemon Generator")

    backend_url = get_backend_url()
    backend_url = st.text_input("Backend URL", value=backend_url)

    steps = st.slider("Sampling steps", min_value=1, max_value=300, value=50, step=1)
    if st.button("Generate"):
        with st.spinner("Generating image"):
            image_bytes = request_image(backend_url, steps)
        st.image(image_bytes)


if __name__ == "__main__":
    main()
