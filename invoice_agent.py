import json
import re
from datetime import date, datetime
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

EXTRACTION_SCHEMA = """
{
  "isInvoice": true or false,
  "confidence": 0-100,
  "confidenceReason": "brief explanation",
  "invoiceNumber": "string or null",
  "vendor": "string or null",
  "vendorAddress": "string or null",
  "billTo": "string or null",
  "date": "YYYY-MM-DD or null",
  "dueDate": "YYYY-MM-DD or null",
  "paymentTerms": "string or null",
  "currency": "USD | EUR | GBP | INR | etc.",
  "subtotal": number or null,
  "taxAmount": number or null,
  "taxRate": number or null,
  "discount": number or null,
  "total": number or null,
  "lineItems": [
    {
      "description": "string",
      "qty": number,
      "unitPrice": number,
      "lineTotal": number
    }
  ],
  "notes": "any additional observations or null"
}
"""


def _call_groq(client: Groq, system: str, user: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _clean_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


def classify_and_extract(client: Groq, text: str, filename: str) -> dict:
    system = (
        "You are an expert invoice parsing agent. "
        "Analyze the document and respond ONLY with a valid JSON object matching the schema. "
        "No markdown, no explanation, no backticks — raw JSON only."
    )
    user = f"""Document filename: {filename}

Document content:
{text[:5000]}

Extract all invoice data and respond with this exact JSON schema:
{EXTRACTION_SCHEMA}

Rules:
- confidence: 90-100 if clearly an invoice, 70-89 if likely, 40-69 if uncertain, 0-39 if not an invoice
- If lineItems are not explicitly listed but a total exists, return an empty array []
- All monetary values as plain numbers (no currency symbols)
- Dates in YYYY-MM-DD format only
- If a field is not found, use null
"""
    raw = _call_groq(client, system, user)
    try:
        data = _clean_json(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    data["_rawText"] = text
    data["_filename"] = filename
    return data


def validate_math(inv: dict) -> dict:
    flags = []
    details = {}

    line_items = inv.get("lineItems") or []
    subtotal   = inv.get("subtotal")
    tax_amount = inv.get("taxAmount")
    total      = inv.get("total")
    tax_rate   = inv.get("taxRate")

    # Line item
    item_errors = []
    computed_subtotal_from_lines = 0.0
    for i, item in enumerate(line_items):
        qty   = item.get("qty") or 1
        price = item.get("unitPrice") or 0
        listed_total = item.get("lineTotal") or 0
        expected = round(qty * price, 2)
        computed_subtotal_from_lines += expected
        if listed_total and abs(expected - listed_total) > 0.02:
            item_errors.append({
                "line": i + 1,
                "description": item.get("description", ""),
                "expected": expected,
                "listed": listed_total,
                "diff": round(listed_total - expected, 2),
            })
    details["lineItemErrors"] = item_errors
    if item_errors:
        flags.append({"type": "error", "msg": f"{len(item_errors)} line item(s) have arithmetic errors"})

    # subtotal = sum of line items
    if line_items and subtotal is not None:
        diff = round(abs(computed_subtotal_from_lines - subtotal), 2)
        details["computedSubtotal"] = round(computed_subtotal_from_lines, 2)
        if diff > 0.05:
            flags.append({"type": "error", "msg": f"Subtotal mismatch: lines sum to {computed_subtotal_from_lines:.2f}, listed {subtotal:.2f}"})
        else:
            flags.append({"type": "ok", "msg": "Line items sum matches subtotal"})

    # subtotal + tax = total
    if subtotal is not None and tax_amount is not None and total is not None:
        expected_total = round(subtotal + tax_amount, 2)
        diff = abs(expected_total - total)
        details["expectedTotal"] = expected_total
        if diff > 0.05:
            flags.append({"type": "error", "msg": f"Total mismatch: {subtotal:.2f} + {tax_amount:.2f} tax ≠ {total:.2f}"})
        else:
            flags.append({"type": "ok", "msg": "Subtotal + tax matches total"})

    # back-calculate tax rate
    if subtotal and subtotal > 0 and tax_amount is not None:
        back_calc_rate = round((tax_amount / subtotal) * 100, 2)
        details["backCalcTaxRate"] = back_calc_rate
        if tax_rate and abs(back_calc_rate - tax_rate) > 0.5:
            flags.append({"type": "warn", "msg": f"Tax rate stated as {tax_rate}% but back-calculated as {back_calc_rate}%"})
        else:
            flags.append({"type": "ok", "msg": f"Tax rate back-calculated: {back_calc_rate}%"})

    # round number heuristic
    if total is not None and total > 0:
        if total == int(total) and total >= 1000:
            flags.append({"type": "warn", "msg": f"Round number total ({total:,.0f}) — worth verifying"})
            details["roundNumber"] = True
        else:
            details["roundNumber"] = False

    # cost per unit stats
    if line_items:
        prices = [it.get("unitPrice") for it in line_items if it.get("unitPrice")]
        if prices:
            details["avgUnitPrice"] = round(sum(prices) / len(prices), 2)
            details["minUnitPrice"] = round(min(prices), 2)
            details["maxUnitPrice"] = round(max(prices), 2)

    details["flags"] = flags
    return details


def financial_analysis(inv: dict) -> dict:
    result = {}
    today_date = date.today()

    # days until due
    due_str = inv.get("dueDate")
    if due_str:
        try:
            due = datetime.strptime(due_str, "%Y-%m-%d").date()
            days_left = (due - today_date).days
            result["daysUntilDue"] = days_left
            if days_left < 0:
                result["urgency"] = "overdue"
                result["urgencyLabel"] = f"Overdue by {abs(days_left)} days"
            elif days_left <= 3:
                result["urgency"] = "critical"
                result["urgencyLabel"] = f"Due in {days_left} day(s)"
            elif days_left <= 7:
                result["urgency"] = "high"
                result["urgencyLabel"] = f"Due in {days_left} days"
            elif days_left <= 14:
                result["urgency"] = "medium"
                result["urgencyLabel"] = f"Due in {days_left} days"
            else:
                result["urgency"] = "low"
                result["urgencyLabel"] = f"Due in {days_left} days"
        except Exception:
            result["daysUntilDue"] = None

    # early payment discount
    terms = inv.get("paymentTerms") or ""
    match = re.search(r'(\d+(?:\.\d+)?)/(\d+)\s*[Nn]et\s*(\d+)', terms)
    if match:
        disc_pct  = float(match.group(1))
        disc_days = int(match.group(2))
        net_days  = int(match.group(3))
        total     = inv.get("total") or 0
        savings   = round(total * disc_pct / 100, 2)
        result["earlyPayDiscount"] = {
            "discountPct":   disc_pct,
            "discountDays":  disc_days,
            "netDays":       net_days,
            "savingsAmount": savings,
            "label": f"Pay within {disc_days} days to save {disc_pct}% (${savings:,.2f})"
        }

    return result


def check_duplicate(current: dict, all_invoices: list) -> dict:
    result = {"isDuplicate": False, "matchedWith": None}
    vendor  = (current.get("vendor") or "").lower().strip()
    total   = current.get("total")
    date_str = current.get("date")

    if not vendor or total is None:
        return result

    try:
        cur_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    except Exception:
        cur_date = None

    for prev in all_invoices:
        if prev is current:
            continue
        prev_vendor = (prev.get("vendor") or "").lower().strip()
        prev_total  = prev.get("total")
        prev_date_str = prev.get("date")

        if prev_vendor != vendor or prev_total != total:
            continue

        try:
            prev_date = datetime.strptime(prev_date_str, "%Y-%m-%d").date() if prev_date_str else None
        except Exception:
            prev_date = None

        if cur_date and prev_date:
            if abs((cur_date - prev_date).days) <= 7:
                result["isDuplicate"] = True
                result["matchedWith"] = prev.get("_filename", "unknown")
                break
        elif cur_date is None and prev_date is None:
            result["isDuplicate"] = True
            result["matchedWith"] = prev.get("_filename", "unknown")
            break

    return result


def process_invoice(client: Groq, text: str, filename: str, all_invoices: list) -> dict:
    inv = classify_and_extract(client, text, filename)

    if inv.get("isInvoice") and inv.get("confidence", 0) >= 70:
        inv["_validation"] = validate_math(inv)
        inv["_financial"]  = financial_analysis(inv)
        inv["_duplicate"]  = check_duplicate(inv, all_invoices)
        inv["_status"]     = "valid"
    elif inv.get("confidence", 0) >= 40:
        inv["_status"]     = "review"
        inv["_validation"] = {}
        inv["_financial"]  = {}
        inv["_duplicate"]  = {}
    else:
        inv["_status"]     = "invalid"
        inv["_validation"] = {}
        inv["_financial"]  = {}
        inv["_duplicate"]  = {}

    return inv
