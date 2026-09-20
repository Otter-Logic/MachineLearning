using System.Globalization;
using System.Text;

namespace OtterLogic.MachineLearning.Data;

/// <summary>
/// Just enough CSV to write a table and read it back: RFC 4180 quoting, and numbers
/// that survive the trip.
/// <para>
/// Hand-rolled because what is needed is forty lines and the alternative is the
/// first package reference in a repo that has none. The two things a casual
/// implementation gets wrong are both handled here: a field holding a comma, quote
/// or line break is quoted, and numbers are written with the invariant culture — on
/// a machine set to German, <c>ToString()</c> writes 1,5 and splits one value into
/// two columns.
/// </para>
/// </summary>
internal static class Csv
{
    /// <summary>
    /// A number as text that parses back to the identical double. "R" rather than a
    /// fixed precision, so a feature written and read is bit-for-bit the feature
    /// that was computed, and a model sees the same input either side of the file.
    /// </summary>
    internal static string Number(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    internal static bool TryNumber(string text, out double value)
        => double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value)
            && double.IsFinite(value);

    internal static void WriteRow(TextWriter writer, IEnumerable<string> fields)
    {
        bool first = true;
        foreach (var field in fields)
        {
            if (!first)
                writer.Write(',');
            first = false;

            if (field.AsSpan().IndexOfAny(",\"\r\n") >= 0 || field != field.Trim())
            {
                writer.Write('"');
                writer.Write(field.Replace("\"", "\"\""));
                writer.Write('"');
            }
            else
            {
                writer.Write(field);
            }
        }

        // Always \n, whatever the platform: the same rows then produce the same
        // bytes on every machine, which is what makes a rewrite a no-op to git or
        // to a sync client watching the folder.
        writer.Write('\n');
    }

    /// <summary>Every record in the text. Blank lines are skipped; a quoted field may span lines.</summary>
    internal static List<string[]> Read(TextReader reader)
    {
        var records = new List<string[]>();
        var fields = new List<string>();
        var field = new StringBuilder();
        bool quoted = false;
        bool wasQuoted = false;

        void EndField()
        {
            fields.Add(field.ToString());
            field.Clear();
            wasQuoted = false;
        }

        void EndRecord()
        {
            // A line with nothing on it is not a record with one empty field.
            if (fields.Count == 0 && field.Length == 0 && !wasQuoted)
                return;

            EndField();
            records.Add(fields.ToArray());
            fields.Clear();
        }

        int read;
        while ((read = reader.Read()) >= 0)
        {
            char c = (char)read;

            if (quoted)
            {
                if (c != '"')
                    field.Append(c);
                else if (reader.Peek() == '"')
                {
                    field.Append('"');
                    reader.Read();
                }
                else
                    quoted = false;

                continue;
            }

            switch (c)
            {
                case '"' when field.Length == 0:
                    quoted = true;
                    wasQuoted = true;
                    break;
                case ',':
                    EndField();
                    break;
                case '\r':
                    break;
                case '\n':
                    EndRecord();
                    break;
                default:
                    field.Append(c);
                    break;
            }
        }

        if (quoted)
            throw new InvalidDataException("A quoted field is never closed.");

        EndRecord();
        return records;
    }
}
