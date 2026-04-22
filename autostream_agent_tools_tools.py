"""
tools.py
--------
Tool definitions used by the AutoStream agent.
Currently contains the mock lead capture function.
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead into a CRM / backend system.

    Args:
        name:     Full name of the prospect.
        email:    Email address of the prospect.
        platform: Content platform they primarily use (e.g. YouTube).

    Returns:
        A dict confirming the lead was captured.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    lead = {
        "status": "success",
        "lead_id": f"LEAD-{abs(hash(email)) % 100000:05d}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "assigned_plan": "Pro Plan (trial)",
    }

    # Simulate console output as required by the assignment
    print(f"\n{'='*55}")
    print(f"  ✅  Lead captured successfully!")
    print(f"  Name    : {name}")
    print(f"  Email   : {email}")
    print(f"  Platform: {platform}")
    print(f"  Lead ID : {lead['lead_id']}")
    print(f"  Time    : {timestamp}")
    print(f"{'='*55}\n")

    return lead
