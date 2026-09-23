using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// Finds the private Python the trainer runs in, and installs one when there is none.
/// <para>
/// Two places, in order. The environment variable <see cref="EnvironmentVariable"/>
/// exists for development: point it at a checkout's <c>.venv</c> with the trainer
/// package installed into it, and training works before any bundle exists. The
/// install folder under <c>%LOCALAPPDATA%</c> is where a bundle unpacks to — one
/// folder per version, so a trainer update never touches the one in use.
/// </para>
/// <para>
/// A bundle is a zip holding a standalone Python with the trainer and its wheels
/// installed, built by <c>python/trainer/build-bundle.ps1</c> and attached to a
/// release of the MachineLearning repo beside a <c>trainer-manifest.json</c> naming
/// it. <see cref="InstallAsync"/> reads the manifest, downloads the zip, checks
/// its hash and unpacks it; <see cref="InstallFromFile"/> takes a zip somebody
/// already has, which is the answer to a proxy that blocks GitHub. No admin
/// rights, no PATH change, nothing outside the one folder.
/// </para>
/// </summary>
public static class TrainerRuntime
{
    /// <summary>
    /// A path to a <c>python.exe</c>, or to a folder holding one directly or under
    /// <c>Scripts\</c> (a virtual environment). Development only.
    /// </summary>
    public const string EnvironmentVariable = "OTTERLOGIC_TRAINER";

    /// <summary>Where the current bundle is described: version, download URL, SHA-256 and size.</summary>
    public const string ManifestUrl =
        "https://github.com/Otter-Logic/MachineLearning/releases/latest/download/trainer-manifest.json";

    /// <summary>The file inside a bundle that says what it is.</summary>
    public const string BundleDescriptor = "bundle.json";

    /// <summary>
    /// A folder to use instead of the real <see cref="InstallRoot"/>. Null in the
    /// plug-in; the tests point it at a temporary folder so an install test never
    /// writes into, or reads from, the folder a user's real runtime lives in.
    /// </summary>
    internal static string? InstallRootOverride { get; set; }

    /// <summary>Where bundles unpack: one subfolder per version, each holding a <c>python.exe</c>.</summary>
    public static string InstallRoot
        => InstallRootOverride
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OtterLogic", "trainer");

    /// <summary>The Python to run the trainer with, or null when there is none anywhere it looks.</summary>
    public static string? Find()
    {
        string? fromEnvironment = FromEnvironment();
        if (fromEnvironment is not null)
            return fromEnvironment;

        return FromInstallRoot();
    }

    /// <summary>
    /// Where it looked and what it found, for the message a user sees when training
    /// cannot run. Both places are named whether or not something was there, so
    /// the fix is in the message.
    /// </summary>
    public static string Describe()
    {
        string? variable = Environment.GetEnvironmentVariable(EnvironmentVariable);
        string? fromEnvironment = FromEnvironment();
        string? installed = FromInstallRoot();

        string environment = variable is null
            ? $"{EnvironmentVariable} is not set."
            : fromEnvironment is null
                ? $"{EnvironmentVariable} is '{variable}', but there is no python.exe there."
                : $"{EnvironmentVariable} points at '{fromEnvironment}'.";

        string install = installed is null
            ? $"Nothing is installed under '{InstallRoot}'."
            : $"'{installed}' is installed.";

        return environment + " " + install;
    }

    /// <summary>
    /// Downloads the current bundle and installs it.
    /// </summary>
    /// <param name="progress">Told what is happening, in sentences a user can read.</param>
    /// <param name="cancellation">Stops the download or the unpack; nothing half-done is left installed.</param>
    /// <param name="manifestUrl">Where to read the manifest; the release's by default.</param>
    /// <returns>The installed <c>python.exe</c>.</returns>
    /// <exception cref="InvalidOperationException">The manifest or the bundle is not what it should be; the message says what.</exception>
    /// <exception cref="HttpRequestException">The download failed.</exception>
    public static async Task<string> InstallAsync(
        IProgress<string>? progress = null, CancellationToken cancellation = default, string manifestUrl = ManifestUrl)
    {
        using var http = new HttpClient();
        http.Timeout = TimeSpan.FromMinutes(30);

        progress?.Report("Reading the release manifest.");
        TrainerBundle bundle;
        using (var response = await http.GetAsync(manifestUrl, cancellation).ConfigureAwait(false))
        {
            response.EnsureSuccessStatusCode();
            string json = await response.Content.ReadAsStringAsync(cancellation).ConfigureAwait(false);
            bundle = TrainerBundle.FromJson(json);
        }

        string existing = Path.Combine(InstallRoot, bundle.Version);
        if (PythonIn(existing) is { } already)
        {
            progress?.Report($"Version {bundle.Version} is already installed.");
            return already;
        }

        string download = Path.Combine(Path.GetTempPath(), "OtterLogic", "trainer-download",
            $"otterlogic-trainer-{bundle.Version}-{Guid.NewGuid():N}.zip");
        Directory.CreateDirectory(Path.GetDirectoryName(download)!);

        try
        {
            progress?.Report($"Downloading the training runtime {bundle.Version} ({bundle.Bytes / (1024.0 * 1024.0):0} MB).");
            using (var response = await http.GetAsync(bundle.Url, HttpCompletionOption.ResponseHeadersRead, cancellation).ConfigureAwait(false))
            {
                response.EnsureSuccessStatusCode();
                await using var source = await response.Content.ReadAsStreamAsync(cancellation).ConfigureAwait(false);
                await using var target = File.Create(download);
                await CopyWithProgress(source, target, bundle.Bytes, progress, cancellation).ConfigureAwait(false);
            }

            progress?.Report("Checking the download.");
            string actual = Sha256Of(download);
            if (!string.Equals(actual, bundle.Sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    "The downloaded runtime does not match the manifest's hash, so it was not installed. "
                    + "Try again; if it keeps happening, the release is damaged.");

            return Unpack(download, bundle.Version, progress, cancellation);
        }
        finally
        {
            try { File.Delete(download); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>
    /// Installs a bundle somebody already has as a file — downloaded once by IT and
    /// put on a share, for machines that cannot reach GitHub.
    /// </summary>
    /// <param name="bundleZip">The zip. Its version is read from the <c>bundle.json</c> inside it.</param>
    /// <param name="progress">Told what is happening.</param>
    /// <param name="cancellation">Stops the unpack; nothing half-done is left installed.</param>
    /// <returns>The installed <c>python.exe</c>.</returns>
    /// <exception cref="InvalidOperationException">The file is not a bundle; the message says what.</exception>
    public static string InstallFromFile(string bundleZip, IProgress<string>? progress = null, CancellationToken cancellation = default)
    {
        if (string.IsNullOrWhiteSpace(bundleZip) || !File.Exists(bundleZip))
            throw new InvalidOperationException($"There is no file at '{bundleZip}'.");

        string version = DescriptorIn(bundleZip).Version;
        return Unpack(bundleZip, version, progress, cancellation);
    }

    /// <summary>
    /// Unpacks into <c>InstallRoot/&lt;version&gt;</c>, by way of a sibling folder that is
    /// renamed into place at the end — so a cancelled or failed unpack leaves nothing
    /// <see cref="Find"/> could mistake for an install.
    /// </summary>
    private static string Unpack(string bundleZip, string version, IProgress<string>? progress, CancellationToken cancellation)
    {
        var descriptor = DescriptorIn(bundleZip);
        if (!string.Equals(descriptor.Version, version, StringComparison.Ordinal))
            throw new InvalidOperationException(
                $"The bundle says it is version {descriptor.Version} and the manifest says {version}. It was not installed.");

        Directory.CreateDirectory(InstallRoot);
        string final = Path.Combine(InstallRoot, version);
        string staging = final + ".installing";
        if (Directory.Exists(staging))
            Directory.Delete(staging, recursive: true);

        try
        {
            progress?.Report($"Unpacking version {version}.");
            using (var archive = ZipFile.OpenRead(bundleZip))
            {
                string root = Path.GetFullPath(staging) + Path.DirectorySeparatorChar;
                foreach (var entry in archive.Entries)
                {
                    cancellation.ThrowIfCancellationRequested();

                    // A zip may name a path that climbs out of its folder; nothing
                    // from a download is written anywhere but under the staging folder.
                    string destination = Path.GetFullPath(Path.Combine(staging, entry.FullName));
                    if (!destination.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException(
                            $"The bundle holds a path outside its folder ('{entry.FullName}'), so it was not installed.");

                    // Python's zipfile writes forward slashes; a zip repacked with
                    // Windows tools writes backslashes. Either ends a folder entry.
                    if (entry.FullName.EndsWith('/') || entry.FullName.EndsWith('\\'))
                    {
                        Directory.CreateDirectory(destination);
                        continue;
                    }

                    Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                    entry.ExtractToFile(destination, overwrite: true);
                }
            }

            string? python = PythonIn(Path.Combine(staging, descriptor.Python)) ?? PythonIn(staging);
            if (python is null)
                throw new InvalidOperationException(
                    $"The bundle unpacked but holds no python.exe at '{descriptor.Python}', so it was not installed.");

            if (Directory.Exists(final))
                Directory.Delete(final, recursive: true);
            Directory.Move(staging, final);
        }
        catch
        {
            if (Directory.Exists(staging))
            {
                try { Directory.Delete(staging, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }

            throw;
        }

        string installed = PythonIn(Path.Combine(final, descriptor.Python)) ?? PythonIn(final)!;
        progress?.Report($"Installed version {version} at '{final}'.");
        return installed;
    }

    private static BundleDescriptorRecord DescriptorIn(string bundleZip)
    {
        try
        {
            using var archive = ZipFile.OpenRead(bundleZip);
            var entry = archive.Entries.FirstOrDefault(e =>
                string.Equals(e.FullName.TrimStart('/'), BundleDescriptor, StringComparison.OrdinalIgnoreCase));
            if (entry is null)
                throw new InvalidOperationException(
                    $"'{bundleZip}' is not a training runtime bundle: it has no {BundleDescriptor} at its root.");

            using var reader = new StreamReader(entry.Open());
            return BundleDescriptorRecord.FromJson(reader.ReadToEnd());
        }
        catch (InvalidDataException ex)
        {
            throw new InvalidOperationException($"'{bundleZip}' is not a zip file: {ex.Message}", ex);
        }
    }

    private static async Task CopyWithProgress(
        Stream source, Stream target, long expected, IProgress<string>? progress, CancellationToken cancellation)
    {
        var buffer = new byte[1 << 16];
        long copied = 0;
        int lastTenth = -1;
        int read;
        while ((read = await source.ReadAsync(buffer, cancellation).ConfigureAwait(false)) > 0)
        {
            await target.WriteAsync(buffer.AsMemory(0, read), cancellation).ConfigureAwait(false);
            copied += read;
            if (expected > 0)
            {
                int tenth = (int)(copied * 10 / expected);
                if (tenth != lastTenth)
                {
                    lastTenth = tenth;
                    progress?.Report($"Downloading: {tenth * 10}%.");
                }
            }
        }
    }

    private static string Sha256Of(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    private static string? FromEnvironment()
    {
        string? value = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return PythonIn(value);
    }

    private static string? FromInstallRoot()
    {
        if (!Directory.Exists(InstallRoot))
            return null;

        // Newest version first. Ordinal on the folder name is wrong for "0.10" vs
        // "0.9", so parse what parses and fall back to the name for what does not.
        // A folder still being unpacked is named so that it is never a candidate.
        return Directory.EnumerateDirectories(InstallRoot)
            .Where(folder => !folder.EndsWith(".installing", StringComparison.OrdinalIgnoreCase))
            .Select(folder => (folder, version: Version.TryParse(Path.GetFileName(folder), out var v) ? v : new Version(0, 0)))
            .OrderByDescending(p => p.version)
            .ThenByDescending(p => p.folder, StringComparer.Ordinal)
            .Select(p => PythonIn(p.folder))
            .FirstOrDefault(python => python is not null);
    }

    /// <summary>The interpreter a path names, whichever of the three shapes it is.</summary>
    private static string? PythonIn(string path)
    {
        if (File.Exists(path))
            return Path.GetFullPath(path);

        if (!Directory.Exists(path))
            return null;

        foreach (var candidate in new[] { "python.exe", Path.Combine("Scripts", "python.exe"), "python", Path.Combine("bin", "python") })
        {
            string full = Path.Combine(path, candidate);
            if (File.Exists(full))
                return Path.GetFullPath(full);
        }

        return null;
    }

    /// <summary>The <c>bundle.json</c> at a bundle's root, written by the bundle build.</summary>
    private sealed record BundleDescriptorRecord
    {
        public required string Version { get; init; }

        /// <summary>The interpreter, relative to the bundle root: <c>python.exe</c> for a standalone Python at the root.</summary>
        public string Python { get; init; } = "python.exe";

        private static readonly JsonSerializerOptions Json = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
        };

        public static BundleDescriptorRecord FromJson(string json)
        {
            BundleDescriptorRecord? record;
            try
            {
                record = JsonSerializer.Deserialize<BundleDescriptorRecord>(json, Json);
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException($"The bundle's {BundleDescriptor} could not be read: {ex.Message}", ex);
            }

            if (record is null || string.IsNullOrWhiteSpace(record.Version))
                throw new InvalidOperationException($"The bundle's {BundleDescriptor} names no version.");

            return record;
        }
    }
}

/// <summary>
/// What the release manifest says about the current bundle.
/// </summary>
public sealed record TrainerBundle
{
    /// <summary>The trainer version, which names the install folder.</summary>
    public required string Version { get; init; }

    /// <summary>Where the zip is.</summary>
    public required string Url { get; init; }

    /// <summary>SHA-256 of the zip, hex. Checked before anything is unpacked.</summary>
    public required string Sha256 { get; init; }

    /// <summary>Size of the zip, for the progress message. Zero when unknown.</summary>
    public long Bytes { get; init; }

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    public string ToJson() => JsonSerializer.Serialize(this, Json);

    public static TrainerBundle FromJson(string json)
    {
        TrainerBundle? bundle;
        try
        {
            bundle = JsonSerializer.Deserialize<TrainerBundle>(json, Json);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"The release manifest could not be read: {ex.Message}", ex);
        }

        if (bundle is null || string.IsNullOrWhiteSpace(bundle.Version) || string.IsNullOrWhiteSpace(bundle.Url)
            || string.IsNullOrWhiteSpace(bundle.Sha256))
            throw new InvalidOperationException("The release manifest is missing its version, URL or hash.");

        return bundle;
    }
}
