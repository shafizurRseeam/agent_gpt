import requests


def get_form_fields(url):
    """
    GET <url>/fields -> returns list of fields the service's booking form requires.
    Each field: {field, label, type, necessary}
    """
    fields_url = url.rstrip("/") + "/fields"
    try:
        r = requests.get(fields_url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data.get("fields", [])
    except Exception as e:
        return {"error": str(e)}


def submit_form(url, data):
    """
    POST <url>/book with data as JSON.
    Returns the service confirmation response.
    """
    book_url = url.rstrip("/") + "/book"
    try:
        r = requests.post(book_url, json=data, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
