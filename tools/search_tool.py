"""
Mock search tool. Returns local service providers with their mock booking URLs.
In production these would come from a real search API.
"""

MOCK_BASE = "http://localhost:8000"

PROVIDERS = {
    "medical": [
        {
            "name": "Rochester General Clinic",
            "address": "1425 Portland Ave, Rochester NY 14621",
            "distance_miles": 1.0,
            "url": f"{MOCK_BASE}/medical/rochester-general"
        },
        {
            "name": "Elmwood Family Medicine",
            "address": "500 Elmwood Ave, Rochester NY 14620",
            "distance_miles": 1.5,
            "url": f"{MOCK_BASE}/medical/elmwood-family"
        },
        {
            "name": "Highland Primary Care",
            "address": "1000 South Ave, Rochester NY 14620",
            "distance_miles": 2.0,
            "url": f"{MOCK_BASE}/medical/highland"
        },
    ],
    "dental": [
        {
            "name": "Bright Smile Dental",
            "address": "120 Main St, Rochester NY 14604",
            "distance_miles": 0.8,
            "url": f"{MOCK_BASE}/dental/bright-smile"
        },
        {
            "name": "Rochester Dental Care",
            "address": "88 Monroe Ave, Rochester NY 14607",
            "distance_miles": 1.2,
            "url": f"{MOCK_BASE}/dental/rochester-dental"
        },
        {
            "name": "Lakeview Dentist",
            "address": "310 Lake Ave, Rochester NY 14608",
            "distance_miles": 2.5,
            "url": f"{MOCK_BASE}/dental/lakeview"
        },
    ],
    "restaurant": [
        {
            "name": "Tony's Italian Kitchen",
            "address": "55 Park Ave, Rochester NY 14607",
            "distance_miles": 1.1,
            "url": f"{MOCK_BASE}/restaurant/tonys"
        },
    ],
    "garage": [
        {
            "name": "QuickLube Car Service",
            "address": "201 Ridge Rd, Rochester NY 14621",
            "distance_miles": 1.8,
            "url": f"{MOCK_BASE}/garage/quicklube"
        },
    ],
}


def search_medical(location):
    return PROVIDERS["medical"]


def search_dentists(location):
    return PROVIDERS["dental"]


def search_restaurants(location):
    return PROVIDERS["restaurant"]


def search_garages(location):
    return PROVIDERS["garage"]
