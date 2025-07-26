"""
=====================================================================
HAPI FHIR PUBLIC TEST‑SERVER – FULL CRUD DEMO
Everything you need to:

  PATIENTS
    • create_patient(full_name)
    • get_patient(patient_id)
    • search_patients(name_fragment=None)

  APPOINTMENTS
    • create_appointment(patient_id, start_dt, end_dt, description)
    • list_appointments(patient_id)
    • get_appointment(appointment_id)
    • reschedule_appointment(appointment_id, new_start_dt, new_end_dt)
    • cancel_appointment(appointment_id)

  RESULTS
    • create_diagnostic_report(patient_id, code, text, value, unit)
    • get_diagnostic_report(report_id)
    • get_diagnostic_reports(patient_id)

All requests hit the open R4 sandbox at https://hapi.fhir.org/baseR4
(no API key; data wiped roughly every 24 h).

Author: ChatGPT (o3) — 26 Jul 2025
=====================================================================
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import requests

BASE_URL = "https://hapi.fhir.org/baseR4"
HEADERS  = {"Content-Type": "application/fhir+json",
            "Accept":       "application/fhir+json"}

# ------------------------------------------------------------------ #
#  LOW‑LEVEL HELPERS
# ------------------------------------------------------------------ #
def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"

def _check(resp: requests.Response) -> dict:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text}") from exc
    return resp.json()

# ------------------------------------------------------------------ #
#  PATIENT ENDPOINTS
# ------------------------------------------------------------------ #
def create_patient(full_name: str) -> str:
    family, *given = full_name.split()
    body = {
        "resourceType": "Patient",
        "identifier": [{"system": "urn:uuid", "value": str(uuid.uuid4())}],
        "name": [{"use": "official", "family": family, "given": given}],
        "active": True
    }
    res = _check(requests.post(_url("Patient"), headers=HEADERS, data=json.dumps(body)))
    pid = res["id"]
    print(f"✅ Created Patient/{pid} ({full_name})")
    return pid


def get_patient(patient_id: str) -> dict:
    return _check(requests.get(_url(f"Patient/{patient_id}"), headers=HEADERS))


def search_patients(name_fragment: Optional[str] = None) -> List[Dict]:
    params = {"name": name_fragment} if name_fragment else None
    bundle = _check(requests.get(_url("Patient"), headers=HEADERS, params=params))
    return [e["resource"] for e in bundle.get("entry", [])]

# ------------------------------------------------------------------ #
#  APPOINTMENT ENDPOINTS
# ------------------------------------------------------------------ #
def create_appointment(patient_id: str,
                       start_dt: datetime,
                       end_dt: datetime,
                       description: str = "Exam Appointment") -> str:
    body = {
        "resourceType": "Appointment",
        "status": "booked",
        "description": description,
        "start": start_dt.astimezone(timezone.utc).isoformat(),
        "end":   end_dt.astimezone(timezone.utc).isoformat(),
        "participant": [{
            "actor": {"reference": f"Patient/{patient_id}"},
            "status": "accepted"
        }]
    }
    res = _check(requests.post(_url("Appointment"), headers=HEADERS, data=json.dumps(body)))
    aid = res["id"]
    print(f"✅ Created Appointment/{aid} for Patient/{patient_id}")
    return aid


def get_appointment(appt_id: str) -> dict:
    return _check(requests.get(_url(f"Appointment/{appt_id}"), headers=HEADERS))


def list_appointments(patient_id: str) -> List[Dict]:
    bundle = _check(requests.get(_url("Appointment"),
                                  headers=HEADERS,
                                  params={"actor": f"Patient/{patient_id}"}))
    appts = [e["resource"] for e in bundle.get("entry", [])]
    print(f"📅 Found {len(appts)} appointment(s) for Patient/{patient_id}")
    return appts


def reschedule_appointment(appt_id: str,
                           new_start_dt: datetime,
                           new_end_dt: datetime) -> None:
    appt = get_appointment(appt_id)
    appt["start"] = new_start_dt.astimezone(timezone.utc).isoformat()
    appt["end"]   = new_end_dt.astimezone(timezone.utc).isoformat()
    _check(requests.put(_url(f"Appointment/{appt_id}"),
                        headers=HEADERS,
                        data=json.dumps(appt)))
    print(f"🔄 Rescheduled Appointment/{appt_id}")


def cancel_appointment(appt_id: str) -> None:
    appt = get_appointment(appt_id)
    appt["status"] = "cancelled"
    _check(requests.put(_url(f"Appointment/{appt_id}"),
                        headers=HEADERS,
                        data=json.dumps(appt)))
    print(f"🚫 Cancelled Appointment/{appt_id}")

# ------------------------------------------------------------------ #
#  RESULTS (DiagnosticReport) ENDPOINTS
# ------------------------------------------------------------------ #
def create_diagnostic_report(patient_id: str,
                             code: str,
                             text: str,
                             value: float,
                             unit: str) -> str:
    """
    Create a simple DiagnosticReport + embedded Observation (numeric result).
    """
    obs_id = f"Observation/{str(uuid.uuid4())}"
    now = datetime.now(timezone.utc).isoformat()

    observation = {
        "resourceType": "Observation",
        "id": obs_id.split("/")[1],
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": text}],
                 "text": text},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now,
        "valueQuantity": {"value": value, "unit": unit}
    }

    report_body = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": text}],
                 "text": text},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now,
        "result": [{"reference": obs_id}]
    }

    # Transaction bundle: create both Observation and Report in one call
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {"resource": observation,
             "request": {"method": "POST", "url": "Observation"}},
            {"resource": report_body,
             "request": {"method": "POST", "url": "DiagnosticReport"}}
        ]
    }

    res = _check(requests.post(_url(""), headers=HEADERS, data=json.dumps(bundle)))
    # Response gives locations; find DiagnosticReport ID
    report_entry = next(e for e in res["entry"]
                        if e["response"]["location"].startswith("DiagnosticReport/"))
    report_id = report_entry["response"]["location"].split("/")[1].split("?")[0]
    print(f"🧪 Created DiagnosticReport/{report_id} for Patient/{patient_id}")
    return report_id


def get_diagnostic_report(report_id: str) -> dict:
    return _check(requests.get(_url(f"DiagnosticReport/{report_id}"), headers=HEADERS))


def get_diagnostic_reports(patient_id: str) -> List[Dict]:
    bundle = _check(requests.get(_url("DiagnosticReport"),
                                 headers=HEADERS,
                                 params={"subject": f"Patient/{patient_id}"}))
    reports = [e["resource"] for e in bundle.get("entry", [])]
    print(f"📄 Found {len(reports)} DiagnosticReport(s) for Patient/{patient_id}")
    return reports

# ------------------------------------------------------------------ #
#  QUICK DEMO (python hapi_full_demo.py)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 1) CREATE PATIENT
    pid = create_patient("Alice Testpatient")

    # ------------------------------------------------------------------ #
    # 2) CREATE APPOINTMENT 1 WEEK FROM NOW
    start = datetime.now(timezone.utc) + timedelta(days=7, hours=1)
    end   = start + timedelta(minutes=30)
    appt_id = create_appointment(pid, start, end, "Routine Blood Test")

    # ------------------------------------------------------------------ #
    # 3) RESCHEDULE APPOINTMENT (PUSH 2 HRS)
    reschedule_appointment(appt_id,
                           start + timedelta(hours=2),
                           end   + timedelta(hours=2))

    # ------------------------------------------------------------------ #
    # 4) RECORD A RESULT FOR THAT PATIENT
    report_id = create_diagnostic_report(pid,
                                         code="4548-4",          # example: HbA1c
                                         text="Hemoglobin A1c",
                                         value=5.4,
                                         unit="%")

    # ------------------------------------------------------------------ #
    # 5) FETCH CURRENT APPOINTMENTS + RESULTS
    list_appointments(pid)
    reports = get_diagnostic_reports(pid)
    for r in reports:
        print(json.dumps(r, indent=2))

    
    # 6) show how to retrieve patient details
    print("\n--- Patient Details ---")
    patient = get_patient(pid)             # same as get_patient("48434449")
    print(json.dumps(patient, indent=2))

    # 7) or search by name
    print("\n--- Search by name 'Alice' ---")
    for p in search_patients("Alice"):
        print("Patient/", p["id"], sep="")
