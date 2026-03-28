"""
services/razorpay_client.py  —  RazorpayClient

Drop-in replacement for all Stripe payment logic.

Razorpay flow (different from Stripe hosted checkout):
  1. Backend creates an Order  → returns order_id + amount to frontend
  2. Frontend opens Razorpay checkout widget with that order_id
  3. User pays → Razorpay calls frontend callback with payment_id, order_id, signature
  4. Frontend sends those three back to /api/verify-checkout
  5. Backend verifies HMAC signature → confirms payment is genuine

Two payment types (mirrors old Stripe checkout_type):
  "chat"  — unlock unlimited messages for a session  (₹100 or configured amount)
  "video" — video memory analysis + unlimited chat   (same amount, different token)

Env vars required:
  RAZORPAY_KEY_ID      — rzp_live_... or rzp_test_...
  RAZORPAY_KEY_SECRET  — your secret key (used for HMAC verification only)

No webhooks needed — HMAC signature verification on /api/verify-checkout is
sufficient for this use case (single page, immediate unlock).
"""

import hashlib
import hmac
import os
import uuid

import razorpay

RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

# Payment amount in paise (100 paise = ₹1). Default ₹100 = 10000 paise.
# Override via env var if you want a different price.
PAYMENT_AMOUNT_PAISE = int(os.environ.get("PAYMENT_AMOUNT_PAISE", "10000"))


def get_razorpay_client() -> razorpay.Client:
    """Return an authenticated Razorpay client."""
    return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


def create_order(checkout_type: str, chronis_session: str) -> dict:
    """
    Create a Razorpay order and return everything the frontend needs to
    open the checkout widget.

    Returns:
        {
            "order_id":   "order_XXXX",
            "amount":     10000,          # paise
            "currency":   "INR",
            "key_id":     "rzp_live_...", # needed by frontend widget
            "type":       "chat"|"video",
            "chronis_session": "...",
        }
    Raises:
        Exception — propagated to the route handler which wraps in HTTPException
    """
    client = get_razorpay_client()

    labels = {
        "chat":  "Continue Conversation — Unlimited Messages",
        "video": "Video Memory Analysis + Unlimited Chat",
    }
    receipt = f"chronis_{checkout_type}_{uuid.uuid4().hex[:8]}"

    order = client.order.create({
        "amount":   PAYMENT_AMOUNT_PAISE,
        "currency": "INR",
        "receipt":  receipt,
        "notes": {
            "chronis_session": chronis_session,
            "type":            checkout_type,
            "product":         labels.get(checkout_type, "Chronis"),
        },
    })

    return {
        "order_id":        order["id"],
        "amount":          order["amount"],
        "currency":        order["currency"],
        "key_id":          RAZORPAY_KEY_ID,
        "type":            checkout_type,
        "chronis_session": chronis_session,
    }


def verify_payment(
    razorpay_order_id:   str,
    razorpay_payment_id: str,
    razorpay_signature:  str,
) -> bool:
    """
    Verify the HMAC-SHA256 signature Razorpay sends after a successful payment.

    Razorpay signs:  order_id + "|" + payment_id
    with your KEY_SECRET and sends it as razorpay_signature.

    Returns True if genuine, False if tampered/invalid.
    """
    if not RAZORPAY_KEY_SECRET:
        return False

    message  = f"{razorpay_order_id}|{razorpay_payment_id}".encode()
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        message,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, razorpay_signature)
