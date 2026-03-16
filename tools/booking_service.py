def book_appointment(doctor):

    print("\n--- ACTUATION ---")
    print(f"Booking appointment with {doctor}")

    return {
        "status": "confirmed",
        "doctor": doctor
    }