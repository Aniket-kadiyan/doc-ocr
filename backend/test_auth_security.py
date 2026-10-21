from __future__ import annotations

import unittest

from auth_security import (
    AuthenticationInputError,
    hash_password,
    normalize_display_name,
    normalize_username,
    verify_password,
    verify_password_and_rehash,
)


class AuthenticationSecurityTests(unittest.TestCase):
    def test_username_is_case_insensitive_but_preserves_display_form(self) -> None:
        self.assertEqual(
            normalize_username("  Inspector.One  "),
            ("Inspector.One", "inspector.one"),
        )

    def test_invalid_username_is_rejected(self) -> None:
        for value in ("ab", "white space", "name/role", "éxample"):
            with self.subTest(value=value):
                with self.assertRaises(AuthenticationInputError):
                    normalize_username(value)

    def test_display_name_is_normalized(self) -> None:
        self.assertEqual(normalize_display_name("  Quality   Admin  "), "Quality Admin")

    def test_password_hash_is_argon2id_and_never_plaintext(self) -> None:
        password = "long development passphrase"
        encoded = hash_password(password, username="administrator")
        self.assertTrue(encoded.startswith("$argon2id$"))
        self.assertNotIn(password, encoded)
        self.assertTrue(verify_password(password, encoded))
        self.assertFalse(verify_password("incorrect passphrase", encoded))

    def test_short_password_and_username_password_are_rejected(self) -> None:
        with self.assertRaisesRegex(AuthenticationInputError, "at least"):
            hash_password("too-short")
        with self.assertRaisesRegex(AuthenticationInputError, "username"):
            hash_password("administrator", username="Administrator")

    def test_malformed_hash_fails_closed(self) -> None:
        self.assertFalse(verify_password("any password here", "not-a-hash"))
        self.assertFalse(
            verify_password_and_rehash("any password here", "not-a-hash").valid
        )


if __name__ == "__main__":
    unittest.main()
