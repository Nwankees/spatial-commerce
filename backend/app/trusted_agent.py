from __future__ import annotations

import base64
import hashlib
import re
import secrets
import time
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .commerce_models import PurchaseIntent, TrustVerification, TrustedAgentEnvelope


_PARAMS = re.compile(
    r'^sig1=\("@authority" "@path" "content-digest"\);created=(\d+);expires=(\d+);'
    r'keyid="([^"]+)";alg="ed25519";nonce="([^"]+)";tag="agent-payer-auth"$'
)
_SIGNATURE = re.compile(r"^sig1=:([A-Za-z0-9+/=]+):$")


class TrustedAgentSigner:
    """Hackathon TAP trust demonstrator using a local Ed25519 development key."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self._private_key = private_key or Ed25519PrivateKey.generate()
        raw = self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        self.key_id = "demo-" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()[:12]).decode().rstrip("=")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private_key.public_key()

    def sign(self, intent: PurchaseIntent, *, now: int | None = None) -> TrustedAgentEnvelope:
        parsed = urlsplit(intent.merchantUrl)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("The merchant checkout URL is not valid.")
        authority = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        created = int(now if now is not None else time.time())
        expires = created + 480
        nonce = secrets.token_urlsafe(18)
        content_digest = "sha-256=:" + base64.b64encode(
            hashlib.sha256(intent.model_dump_json().encode()).digest()
        ).decode("ascii") + ":"
        params = (
            f'sig1=("@authority" "@path" "content-digest");created={created};expires={expires};'
            f'keyid="{self.key_id}";alg="ed25519";nonce="{nonce}";tag="agent-payer-auth"'
        )
        base = _signature_base(authority, path, content_digest, params)
        signature = base64.b64encode(self._private_key.sign(base.encode())).decode("ascii")
        return TrustedAgentEnvelope(
            authority=authority,
            path=path,
            signatureInput=params,
            signature=f"sig1=:{signature}:",
            intentId=intent.id,
            contentDigest=content_digest,
        )


class MerchantTrustedAgentVerifier:
    """Controlled merchant verifier with timestamp, tamper, and replay checks."""

    def __init__(self, keys: dict[str, Ed25519PublicKey]) -> None:
        self._keys = keys
        self._seen_nonces: set[str] = set()

    def verify(
        self,
        envelope: TrustedAgentEnvelope,
        intent: PurchaseIntent | None = None,
        *,
        now: int | None = None,
    ) -> TrustVerification:
        match = _PARAMS.fullmatch(envelope.signatureInput)
        signature_match = _SIGNATURE.fullmatch(envelope.signature)
        if not match or not signature_match:
            return TrustVerification(verified=False, reason="Malformed HTTP message signature.")
        created, expires, key_id, nonce = match.groups()
        created_i, expires_i = int(created), int(expires)
        current = int(now if now is not None else time.time())
        if expires_i <= created_i or expires_i - created_i > 480:
            return TrustVerification(verified=False, keyId=key_id, reason="Invalid signature lifetime.")
        if current < created_i - 30 or current > expires_i:
            return TrustVerification(verified=False, keyId=key_id, reason="Signature is expired or not yet valid.")
        if nonce in self._seen_nonces:
            return TrustVerification(verified=False, keyId=key_id, reason="Replay nonce was already used.")
        key = self._keys.get(key_id)
        if key is None:
            return TrustVerification(verified=False, keyId=key_id, reason="Signing key is not trusted.")
        if intent is not None:
            expected_digest = "sha-256=:" + base64.b64encode(
                hashlib.sha256(intent.model_dump_json().encode()).digest()
            ).decode("ascii") + ":"
            if envelope.intentId != intent.id or envelope.contentDigest != expected_digest:
                return TrustVerification(verified=False, keyId=key_id, reason="Signed purchase intent was tampered with.")
        try:
            key.verify(
                base64.b64decode(signature_match.group(1), validate=True),
                _signature_base(
                    envelope.authority,
                    envelope.path,
                    envelope.contentDigest,
                    envelope.signatureInput,
                ).encode(),
            )
        except (InvalidSignature, ValueError):
            return TrustVerification(verified=False, keyId=key_id, reason="Signature verification failed.")
        self._seen_nonces.add(nonce)
        return TrustVerification(verified=True, keyId=key_id)


def _signature_base(authority: str, path: str, content_digest: str, signature_input: str) -> str:
    params = signature_input.removeprefix("sig1=")
    return (
        f'"@authority": {authority}\n"@path": {path}\n'
        f'"content-digest": {content_digest}\n"@signature-params": {params}'
    )
