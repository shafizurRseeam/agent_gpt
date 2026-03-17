"""
Mock third-party service server.
Run with: uv run mock_services/server.py
Serves on http://localhost:8000

Static services (pre-baked, for reference):
  GET  /<service>/fields  -> JSON list of form fields
  POST /<service>/book    -> JSON booking confirmation
  GET  /<service>/        -> HTML form (for viewing in browser)

Dynamic services (registered at runtime from CLM output):
  POST /services/register          -> register a new service by name+category
  GET  /dynamic/<id>/fields        -> form fields for a registered service
  POST /dynamic/<id>/book          -> book appointment
  GET  /dynamic/<id>/              -> HTML form (for browser viewing)
"""

import re
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Field templates by service category ───────────────────────────────────────
# These define what fields each category of service collects,
# including over-collected (necessary=False) sensitive fields.

FIELD_TEMPLATES = {
    "medical": [
        {"field": "name",             "label": "Full Name",                    "type": "text",   "necessary": True},
        {"field": "phone",            "label": "Phone Number",                 "type": "text",   "necessary": True},
        {"field": "address",          "label": "Home Address",                 "type": "text",   "necessary": True},
        {"field": "insurance",        "label": "Insurance Provider",           "type": "text",   "necessary": True},
        {"field": "appointment_date", "label": "Preferred Appointment Date",   "type": "date",   "necessary": True},
        {"field": "ssn",              "label": "Social Security Number",       "type": "text",   "necessary": False},
        {"field": "dob",              "label": "Date of Birth",                "type": "date",   "necessary": False},
    ],
    "dental": [
        {"field": "name",             "label": "Full Name",                    "type": "text",   "necessary": True},
        {"field": "phone",            "label": "Phone Number",                 "type": "text",   "necessary": True},
        {"field": "address",          "label": "Home Address",                 "type": "text",   "necessary": True},
        {"field": "insurance",        "label": "Insurance Provider",           "type": "text",   "necessary": True},
        {"field": "appointment_date", "label": "Preferred Appointment Date",   "type": "date",   "necessary": True},
        {"field": "ssn",              "label": "Social Security Number",       "type": "text",   "necessary": False},
        {"field": "dob",              "label": "Date of Birth",                "type": "date",   "necessary": False},
    ],
    "restaurant": [
        {"field": "name",             "label": "Reservation Name",             "type": "text",   "necessary": True},
        {"field": "phone",            "label": "Phone Number",                 "type": "text",   "necessary": True},
        {"field": "party_size",       "label": "Party Size",                   "type": "number", "necessary": True},
        {"field": "date",             "label": "Reservation Date",             "type": "date",   "necessary": True},
        {"field": "time",             "label": "Reservation Time",             "type": "time",   "necessary": True},
        {"field": "credit_card",      "label": "Credit Card Number",           "type": "text",   "necessary": False},
    ],
    "garage": [
        {"field": "name",             "label": "Full Name",                    "type": "text",   "necessary": True},
        {"field": "phone",            "label": "Phone Number",                 "type": "text",   "necessary": True},
        {"field": "car_make_model",   "label": "Car Make & Model",             "type": "text",   "necessary": True},
        {"field": "appointment_date", "label": "Preferred Date",               "type": "date",   "necessary": True},
        {"field": "driver_license",   "label": "Driver's License Number",      "type": "text",   "necessary": False},
    ],
    "general": [
        {"field": "name",             "label": "Full Name",                    "type": "text",   "necessary": True},
        {"field": "phone",            "label": "Phone Number",                 "type": "text",   "necessary": True},
        {"field": "address",          "label": "Home Address",                 "type": "text",   "necessary": True},
        {"field": "appointment_date", "label": "Preferred Date",               "type": "date",   "necessary": True},
        {"field": "ssn",              "label": "Social Security Number",       "type": "text",   "necessary": False},
    ],
}

# Registry for dynamically created services (populated at runtime by the agent)
DYNAMIC_SERVICES = {}


def _slugify(name):
    """Convert a service name to a URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


# ── Dynamic service registration ───────────────────────────────────────────────

@app.route("/services/register", methods=["POST"])
def register_service():
    """
    Register a service dynamically from CLM output.
    Body: {name, address, phone, category}
    Returns: {status, slug, url}
    """
    data = request.get_json()
    name     = data.get("name", "Unknown Service")
    address  = data.get("address", "")
    phone    = data.get("phone", "")
    category = data.get("category", "general")

    slug = _slugify(name)
    fields = [dict(f) for f in FIELD_TEMPLATES.get(category, FIELD_TEMPLATES["general"])]

    DYNAMIC_SERVICES[slug] = {
        "name":     name,
        "address":  address,
        "phone":    phone,
        "fields":   fields,
        "category": category,
    }

    return jsonify({
        "status": "registered",
        "slug":   slug,
        "url":    f"http://localhost:8000/dynamic/{slug}",
    })


@app.route("/services/list", methods=["GET"])
def list_dynamic():
    return jsonify({slug: svc["name"] for slug, svc in DYNAMIC_SERVICES.items()})


# ── Dynamic service routes ─────────────────────────────────────────────────────

@app.route("/dynamic/<service_id>/")
def dynamic_form(service_id):
    svc = DYNAMIC_SERVICES.get(service_id)
    if not svc:
        return f"<h2>Service '{service_id}' not found</h2>", 404
    return render_template_string(FORM_TEMPLATE, service=svc, service_key=f"dynamic/{service_id}")


@app.route("/dynamic/<service_id>/fields")
def dynamic_fields(service_id):
    svc = DYNAMIC_SERVICES.get(service_id)
    if not svc:
        return jsonify({"error": f"Service '{service_id}' not registered"}), 404
    return jsonify({"service": svc["name"], "fields": svc["fields"]})


@app.route("/dynamic/<service_id>/book", methods=["POST"])
def dynamic_book(service_id):
    svc = DYNAMIC_SERVICES.get(service_id)
    if not svc:
        return jsonify({"error": f"Service '{service_id}' not registered"}), 404

    data = request.get_json(silent=True) or request.form.to_dict()
    submitted = [
        {
            "field":     f["field"],
            "label":     f["label"],
            "value":     data.get(f["field"], ""),
            "necessary": f["necessary"],
        }
        for f in svc["fields"]
    ]

    if request.is_json:
        return jsonify({
            "status":           "confirmed",
            "service":          svc["name"],
            "submitted_fields": submitted,
        })

    return render_template_string(CONFIRM_TEMPLATE,
                                  service_name=svc["name"],
                                  submitted=submitted)




# ── Service definitions ────────────────────────────────────────────────────────
# necessary=False marks fields that are over-collected (privacy work later)

SERVICES = {
    "medical/rochester-general": {
        "name": "Rochester General Clinic",
        "address": "1425 Portland Ave, Rochester NY 14621",
        "phone": "585-555-0201",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "address",          "label": "Home Address",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
            {"field": "dob",              "label": "Date of Birth",             "type": "date",  "necessary": False},
        ]
    },
    "medical/elmwood-family": {
        "name": "Elmwood Family Medicine",
        "address": "500 Elmwood Ave, Rochester NY 14620",
        "phone": "585-555-0202",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
        ]
    },
    "medical/highland": {
        "name": "Highland Primary Care",
        "address": "1000 South Ave, Rochester NY 14620",
        "phone": "585-555-0203",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "address",          "label": "Home Address",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
            {"field": "credit_card",      "label": "Credit Card (for no-show fee)", "type": "text", "necessary": False},
        ]
    },
    "dental/bright-smile": {
        "name": "Bright Smile Dental",
        "address": "120 Main St, Rochester NY 14604",
        "phone": "585-555-0301",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "address",          "label": "Home Address",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
        ]
    },
    "dental/rochester-dental": {
        "name": "Rochester Dental Care",
        "address": "88 Monroe Ave, Rochester NY 14607",
        "phone": "585-555-0412",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "address",          "label": "Home Address",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
            {"field": "dob",              "label": "Date of Birth",             "type": "date",  "necessary": False},
        ]
    },
    "dental/lakeview": {
        "name": "Lakeview Dentist",
        "address": "310 Lake Ave, Rochester NY 14608",
        "phone": "585-555-0523",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "insurance",        "label": "Insurance Provider",        "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Appointment Date","type": "date",  "necessary": True},
            {"field": "ssn",              "label": "Social Security Number",    "type": "text",  "necessary": False},
        ]
    },
    "restaurant/tonys": {
        "name": "Tony's Italian Kitchen",
        "address": "55 Park Ave, Rochester NY 14607",
        "phone": "585-555-0634",
        "fields": [
            {"field": "name",             "label": "Reservation Name",          "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "party_size",       "label": "Party Size",                "type": "number","necessary": True},
            {"field": "date",             "label": "Reservation Date",          "type": "date",  "necessary": True},
            {"field": "time",             "label": "Reservation Time",          "type": "time",  "necessary": True},
            {"field": "credit_card",      "label": "Credit Card Number",        "type": "text",  "necessary": False},
        ]
    },
    "garage/quicklube": {
        "name": "QuickLube Car Service",
        "address": "201 Ridge Rd, Rochester NY 14621",
        "phone": "585-555-0745",
        "fields": [
            {"field": "name",             "label": "Full Name",                 "type": "text",  "necessary": True},
            {"field": "phone",            "label": "Phone Number",              "type": "text",  "necessary": True},
            {"field": "car_make_model",   "label": "Car Make & Model",          "type": "text",  "necessary": True},
            {"field": "appointment_date", "label": "Preferred Date",            "type": "date",  "necessary": True},
            {"field": "driver_license",   "label": "Driver's License Number",   "type": "text",  "necessary": False},
        ]
    },
}

# ── HTML template ──────────────────────────────────────────────────────────────

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>{{ service.name }} — Book Appointment</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
    h1   { color: #2c3e50; }
    .subtitle { color: #7f8c8d; margin-bottom: 30px; }
    label { display: block; margin-top: 16px; font-weight: bold; font-size: 14px; }
    input { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    .unnecessary { background: #fff3cd; border-color: #ffc107; }
    .unnecessary-label { color: #856404; }
    .tag { font-size: 11px; background: #ffc107; color: #333; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }
    button { margin-top: 24px; padding: 10px 24px; background: #2ecc71; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
    button:hover { background: #27ae60; }
    .note { margin-top: 20px; font-size: 12px; color: #999; }
  </style>
</head>
<body>
  <h1>{{ service.name }}</h1>
  <p class="subtitle">{{ service.address }} &nbsp;|&nbsp; {{ service.phone }}</p>
  <form action="/{{ service_key }}/book" method="POST">
    {% for f in service.fields %}
      {% if f.necessary %}
        <label>{{ f.label }}</label>
        <input type="{{ f.type }}" name="{{ f.field }}" placeholder="{{ f.label }}" required>
      {% else %}
        <label class="unnecessary-label">{{ f.label }} <span class="tag">NOT NECESSARY</span></label>
        <input class="unnecessary" type="{{ f.type }}" name="{{ f.field }}" placeholder="{{ f.label }}">
      {% endif %}
    {% endfor %}
    <br>
    <button type="submit">Book Now</button>
  </form>
  <p class="note">Fields highlighted in yellow are collected by this service but are not required to complete your booking.</p>
</body>
</html>
"""

CONFIRM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Booking Confirmed</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
    h1   { color: #27ae60; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    td, th { padding: 10px; border: 1px solid #ddd; text-align: left; }
    th { background: #f4f4f4; }
    .unnecessary { background: #fff3cd; }
  </style>
</head>
<body>
  <h1>Booking Confirmed</h1>
  <p><strong>Service:</strong> {{ service_name }}</p>
  <h3>Fields Submitted:</h3>
  <table>
    <tr><th>Field</th><th>Value</th><th>Necessary?</th></tr>
    {% for row in submitted %}
    <tr class="{{ 'unnecessary' if not row.necessary else '' }}">
      <td>{{ row.label }}</td>
      <td>{{ row.value }}</td>
      <td>{{ 'Yes' if row.necessary else 'No — over-collected' }}</td>
    </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    links = []
    for key, svc in SERVICES.items():
        links.append(f'<li><a href="/{key}/">{svc["name"]}</a> — <code>/{key}/fields</code></li>')
    dyn_links = []
    for slug, svc in DYNAMIC_SERVICES.items():
        dyn_links.append(f'<li><a href="/dynamic/{slug}/">{svc["name"]}</a> — <code>/dynamic/{slug}/fields</code></li>')
    dyn_section = f"<h2>Dynamic Services (registered at runtime)</h2><ul>{''.join(dyn_links)}</ul>" if dyn_links else ""
    return f"<h2>Static Mock Services</h2><ul>{''.join(links)}</ul>{dyn_section}"


def _make_routes(service_key, service):

    def form_page():
        return render_template_string(FORM_TEMPLATE, service=service, service_key=service_key)

    def fields():
        return jsonify({"service": service["name"], "fields": service["fields"]})

    def book():
        data = request.get_json(silent=True) or request.form.to_dict()
        submitted = []
        for f in service["fields"]:
            submitted.append({
                "field":     f["field"],
                "label":     f["label"],
                "value":     data.get(f["field"], ""),
                "necessary": f["necessary"]
            })

        # If JSON request (from agent), return JSON
        if request.is_json:
            return jsonify({
                "status": "confirmed",
                "service": service["name"],
                "submitted_fields": submitted
            })

        # If browser form POST, return HTML confirmation page
        return render_template_string(CONFIRM_TEMPLATE,
                                      service_name=service["name"],
                                      submitted=submitted)

    # Unique endpoint names per service
    safe = service_key.replace("/", "_")
    app.add_url_rule(f"/{service_key}/",      f"form_{safe}",   form_page,  methods=["GET"])
    app.add_url_rule(f"/{service_key}/fields", f"fields_{safe}", fields,     methods=["GET"])
    app.add_url_rule(f"/{service_key}/book",   f"book_{safe}",   book,       methods=["POST"])


for key, svc in SERVICES.items():
    _make_routes(key, svc)


if __name__ == "__main__":
    print("\n Mock Services running at http://localhost:8000")
    print(" Services available:")
    for key, svc in SERVICES.items():
        print(f"   http://localhost:8000/{key}/")
    app.run(port=8000, debug=False)
