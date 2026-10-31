// Converts a DWG drawing to ASCII DXF for backend/cad_documents.py.
//
// DWG is a closed binary format and Python has no permissively licensed
// reader that keeps a drawing intact, so this small tool uses ACadSharp (MIT)
// to read it and write DXF, which ezdxf then draws to PDF. Model space, every
// paper-space layout and their viewports are carried over unchanged.
//
// Usage: dwg2dxf-acadsharp <input.dwg> <output.dxf>
// Exit codes: 0 converted, 1 the drawing could not be read or written, 2 usage.

using ACadSharp;
using ACadSharp.IO;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: dwg2dxf-acadsharp <input.dwg> <output.dxf>");
    return 2;
}

var (input, output) = (args[0], args[1]);
var warnings = 0;

void Notify(object? sender, NotificationEventArgs e)
{
    // Unsupported objects are reported per entity; the totals are enough.
    if (e.NotificationType is NotificationType.Error or NotificationType.Warning)
    {
        warnings++;
    }
}

try
{
    // Failsafe skips objects it cannot decode instead of rejecting the file.
    var readConfig = new DwgReaderConfiguration { Failsafe = true };
    CadDocument doc = DwgReader.Read(input, readConfig, Notify);
    DxfWriter.Write(output, doc, false, new DxfWriterConfiguration(), Notify);
}
catch (Exception error)
{
    Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
    return 1;
}

if (warnings > 0)
{
    Console.Error.WriteLine($"converted with {warnings} warnings");
}
return 0;
