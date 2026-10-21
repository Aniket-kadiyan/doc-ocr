from __future__ import annotations

import unittest

from pydantic import ValidationError

from client_diagnostics import ClientDiagnosticEvent


class ClientDiagnosticTests(unittest.TestCase):
    def test_free_form_content_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClientDiagnosticEvent.model_validate({
                "kind": "runtime_error", "operation": "window.error",
                "error_type": "TypeError", "message": "private value 12.5",
            })

    def test_invalid_tokens_and_status_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClientDiagnosticEvent.model_validate({
                "kind": "api_error", "operation": "drawing value 12.5",
                "error_type": "ApiRequestError", "http_status": 999,
            })


if __name__ == "__main__":
    unittest.main()
