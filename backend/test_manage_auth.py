from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from auth_security import AuthenticationInputError
from manage_auth import _parser, _prompt_password


class ManageAuthenticationTests(unittest.TestCase):
    def test_create_admin_accepts_no_command_line_password_option(self) -> None:
        parser = _parser()
        arguments = parser.parse_args([
            "user",
            "create-admin",
            "--username",
            "admin",
        ])
        self.assertEqual(arguments.username, "admin")
        self.assertFalse(hasattr(arguments, "password"))

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "user",
                    "create-admin",
                    "--username",
                    "admin",
                    "--password",
                    "must-not-be-supported",
                ])

    def test_interactive_password_confirmation_must_match(self) -> None:
        with patch("manage_auth.getpass", side_effect=["first value", "second value"]):
            with self.assertRaisesRegex(AuthenticationInputError, "do not match"):
                _prompt_password()


if __name__ == "__main__":
    unittest.main()
