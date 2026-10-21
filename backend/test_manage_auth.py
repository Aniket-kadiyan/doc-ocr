from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from auth_security import AuthenticationInputError
from manage_auth import _parser, _prompt_password


class ManageAuthenticationTests(unittest.TestCase):
    def test_create_commands_accept_no_command_line_password_option(self) -> None:
        parser = _parser()
        for action in ("create-user", "create-admin"):
            arguments = parser.parse_args([
                "user",
                action,
                "--username",
                "operator.one",
            ])
            self.assertEqual(arguments.username, "operator.one")
            self.assertFalse(hasattr(arguments, "password"))

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([
                        "user",
                        action,
                        "--username",
                        "operator.one",
                        "--password",
                        "must-not-be-supported",
                    ])

    def test_role_management_commands_accept_username(self) -> None:
        parser = _parser()
        for action in ("grant-admin", "revoke-admin"):
            arguments = parser.parse_args([
                "user",
                action,
                "operator.one",
            ])
            self.assertEqual(arguments.username, "operator.one")
            self.assertEqual(arguments.user_action, action)

    def test_interactive_password_confirmation_must_match(self) -> None:
        with patch("manage_auth.getpass", side_effect=["first value", "second value"]):
            with self.assertRaisesRegex(AuthenticationInputError, "do not match"):
                _prompt_password()


if __name__ == "__main__":
    unittest.main()
