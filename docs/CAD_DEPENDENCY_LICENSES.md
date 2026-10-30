# CAD/DXF/DWG dependency licences

QTracX uses the following components for CAD drawing import.

| Component | Pinned range/version | Licence | Role |
|---|---:|---|---|
| ACadSharp | 3.8.0 | MIT | Read DWG and write an intermediate DXF |
| ezdxf | >=1.3,<2 | MIT | Read and render DXF layouts |
| Matplotlib | >=3.8,<4 | PSF-based, BSD-compatible | Produce vector PDF pages |
| .NET runtime | .NET 10 target | MIT on Linux/macOS; Microsoft .NET Library terms apply to Windows distribution | Run the self-contained DWG converter |

No LibreDWG, ODA File Converter, GPL, AGPL, cloud conversion service, or
proprietary CAD SDK is used by this implementation.

## Distribution obligations

- Preserve the copyright and licence notices for ACadSharp and ezdxf.
- Preserve Matplotlib's licence and copyright notice. If Matplotlib itself is
  modified, include a brief summary of those modifications.
- When distributing a self-contained Windows .NET build, include Microsoft's
  applicable .NET notices and distribute the runtime only as part of QTracX,
  not as a standalone product.
- Include generated third-party notices from the final Python and .NET
  dependency lock/build outputs in the deployment package.

The Windows release workflow copies the following beside the CAD executable:

- `ACADSHARP_NOTICE.md` from the reviewed upstream MIT notice;
- `DOTNET_LICENSE.txt`; and
- `DOTNET_THIRD_PARTY_NOTICES.txt` from the exact .NET runtime used to publish
  the self-contained executable.

This inventory is an engineering compatibility review, not legal advice.
A final customer release should retain the exact notices produced from the
versions actually bundled by the release-readiness workflow.

## Authoritative references

- https://www.nuget.org/packages/ACadSharp/3.8.0
- https://github.com/DomCR/ACadSharp/blob/master/LICENSE
- https://pypi.org/project/ezdxf/
- https://matplotlib.org/stable/project/license.html
- https://dotnet.microsoft.com/en-us/dotnet_library_license.htm
