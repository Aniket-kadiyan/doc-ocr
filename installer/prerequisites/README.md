# Optional offline prerequisite bundle

This folder intentionally contains no third-party installers in source control.
Run `installer\prepare-prerequisites.ps1 -Mode Download` on the release-build
machine to download the versions pinned in
`installer\config\prerequisites.windows-x64.json`, verify published hashes
where available, and create `prerequisites.lock.json`.

The deployment package may then include this folder. The installer prefers a
bundled file whose SHA-256 matches the lock. If a bundle is missing or invalid,
it uses the exact-version WinGet package. The plan always displays the direct
official URL as a manual fallback.

Do not commit downloaded EXE/MSI files. Review third-party redistribution and
license terms before distributing an installer bundle to a customer.
