using System.IO.Compression;
using System.Text;
using OtterLogic.MachineLearning.Training;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// Installing a bundle from a file, against a temporary install root.
/// <para>
/// <see cref="TrainerRuntime.InstallRoot"/> is the folder a user's real runtime
/// lives in, so every test here points <see cref="TrainerRuntime.InstallRootOverride"/>
/// at a folder of its own and clears it after. The class shares a collection with
/// <see cref="TrainerProcessTests"/> so the two never run at once: while the
/// override is set, <see cref="TrainerRuntime.Find"/> would offer the fake
/// <c>python.exe</c> written here to a test that wants a real one.
/// </para>
/// </summary>
[Collection("trainer runtime")]
public class TrainerRuntimeTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "otterlogic-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly string _installRoot;

    public TrainerRuntimeTests()
    {
        Directory.CreateDirectory(_root);
        _installRoot = Path.Combine(_root, "install");
        TrainerRuntime.InstallRootOverride = _installRoot;
    }

    public void Dispose()
    {
        TrainerRuntime.InstallRootOverride = null;
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    /// <summary>A zip holding the entries given, each as its own text.</summary>
    private string Zip(string name, params (string path, string? content)[] entries)
    {
        string zip = Path.Combine(_root, name);
        using var archive = ZipFile.Open(zip, ZipArchiveMode.Create);
        foreach (var (path, content) in entries)
        {
            var entry = archive.CreateEntry(path);
            if (content is null) continue;
            using var writer = new StreamWriter(entry.Open(), new UTF8Encoding(false));
            writer.Write(content);
        }

        return zip;
    }

    private static string Descriptor(string version, string python = "python.exe")
        => $$"""{"version": "{{version}}", "python": "{{python}}"}""";

    [Fact]
    public void InstallsUnderTheVersionAndLeavesNoStagingFolder()
    {
        string zip = Zip("bundle.zip",
            ("bundle.json", Descriptor("0.2.0")),
            ("python.exe", "not really"),
            ("Lib/site-packages/otterlogic_trainer/__init__.py", "__version__ = '0.2.0'"));
        var messages = new List<string>();

        string python = TrainerRuntime.InstallFromFile(zip, new Progress<string>(messages.Add));

        string final = Path.Combine(_installRoot, "0.2.0");
        Assert.Equal(Path.Combine(final, "python.exe"), python);
        Assert.True(File.Exists(python));
        Assert.True(File.Exists(Path.Combine(final, "Lib", "site-packages", "otterlogic_trainer", "__init__.py")));
        Assert.Equal(new[] { "0.2.0" }, Directory.GetDirectories(_installRoot).Select(Path.GetFileName));
        Assert.False(Directory.Exists(final + ".installing"));

        // The install is what the runtime now finds, whatever the environment
        // variable says: Describe names it on its own.
        Assert.Contains($"'{python}' is installed.", TrainerRuntime.Describe());
    }

    [Fact]
    public void FindsThePythonWhereTheDescriptorSaysItIs()
    {
        string zip = Zip("nested.zip",
            ("bundle.json", Descriptor("0.3.0", "python/python.exe")),
            ("python/", null),
            ("python/python.exe", "not really"));

        string python = TrainerRuntime.InstallFromFile(zip);

        Assert.Equal(Path.Combine(_installRoot, "0.3.0", "python", "python.exe"), python);
    }

    [Fact]
    public void InstallingTheSameVersionAgainReplacesIt()
    {
        TrainerRuntime.InstallFromFile(Zip("first.zip", ("bundle.json", Descriptor("0.2.0")), ("python.exe", "one"), ("stale.txt", "x")));
        TrainerRuntime.InstallFromFile(Zip("second.zip", ("bundle.json", Descriptor("0.2.0")), ("python.exe", "two")));

        string final = Path.Combine(_installRoot, "0.2.0");
        Assert.Equal("two", File.ReadAllText(Path.Combine(final, "python.exe")));
        Assert.False(File.Exists(Path.Combine(final, "stale.txt")));
        Assert.False(Directory.Exists(final + ".installing"));
    }

    [Fact]
    public void AZipWithoutADescriptorIsRefused()
    {
        string zip = Zip("plain.zip", ("python.exe", "not really"), ("readme.txt", "no bundle.json here"));

        var ex = Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(zip));

        Assert.Contains(TrainerRuntime.BundleDescriptor, ex.Message);
        Assert.False(Directory.Exists(_installRoot) && Directory.EnumerateFileSystemEntries(_installRoot).Any());
    }

    [Fact]
    public void ADescriptorWithoutAVersionIsRefused()
    {
        string zip = Zip("noversion.zip", ("bundle.json", """{"python": "python.exe"}"""), ("python.exe", "x"));

        var ex = Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(zip));

        Assert.Contains("version", ex.Message);
    }

    [Fact]
    public void ABundleWithoutItsPythonIsRefusedAndRemoved()
    {
        string zip = Zip("nopython.zip", ("bundle.json", Descriptor("0.2.0")), ("readme.txt", "x"));

        var ex = Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(zip));

        Assert.Contains("python.exe", ex.Message);
        Assert.Empty(Directory.GetDirectories(_installRoot));
    }

    [Fact]
    public void APathThatClimbsOutIsRefusedAndNothingIsLeft()
    {
        string zip = Zip("escape.zip",
            ("bundle.json", Descriptor("0.2.0")),
            ("python.exe", "not really"),
            ("../escaped.txt", "should never land"));

        var ex = Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(zip));

        Assert.Contains("outside", ex.Message);
        Assert.Contains("../escaped.txt", ex.Message);
        Assert.False(File.Exists(Path.Combine(_installRoot, "escaped.txt")));
        Assert.False(File.Exists(Path.Combine(_root, "escaped.txt")));
        Assert.Empty(Directory.GetDirectories(_installRoot));
        Assert.DoesNotContain("0.2.0", TrainerRuntime.Describe());
    }

    [Fact]
    public void AFileThatIsNotAZipIsRefused()
    {
        string notZip = Path.Combine(_root, "text.zip");
        File.WriteAllText(notZip, "this is not a zip");

        var ex = Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(notZip));
        Assert.Contains("not a zip", ex.Message);

        Assert.Throws<InvalidOperationException>(() => TrainerRuntime.InstallFromFile(Path.Combine(_root, "missing.zip")));
    }

    [Fact]
    public void CancellationLeavesNothingInstalled()
    {
        string zip = Zip("cancel.zip", ("bundle.json", Descriptor("0.2.0")), ("python.exe", "x"));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        Assert.Throws<OperationCanceledException>(() => TrainerRuntime.InstallFromFile(zip, null, cancelled.Token));

        Assert.Empty(Directory.GetDirectories(_installRoot));
    }

    [Fact]
    public void AFolderStillBeingUnpackedIsNeverFound()
    {
        string half = Path.Combine(_installRoot, "9.9.9.installing");
        Directory.CreateDirectory(half);
        File.WriteAllText(Path.Combine(half, "python.exe"), "x");

        Assert.Contains("Nothing is installed", TrainerRuntime.Describe());

        string installed = TrainerRuntime.InstallFromFile(Zip("b.zip", ("bundle.json", Descriptor("0.2.0")), ("python.exe", "x")));
        Assert.Contains($"'{installed}' is installed.", TrainerRuntime.Describe());
    }

    [Fact]
    public void TheNewestVersionWins()
    {
        TrainerRuntime.InstallFromFile(Zip("a.zip", ("bundle.json", Descriptor("0.9.0")), ("python.exe", "x")));
        TrainerRuntime.InstallFromFile(Zip("b.zip", ("bundle.json", Descriptor("0.10.0")), ("python.exe", "x")));

        // "0.10.0" sorts before "0.9.0" as text; as a version it is newer.
        Assert.Contains(Path.Combine(_installRoot, "0.10.0", "python.exe"), TrainerRuntime.Describe());
    }

    [Fact]
    public void TheManifestRoundTrips()
    {
        var bundle = new TrainerBundle
        {
            Version = "0.2.0",
            Url = "https://github.com/Otter-Logic/MachineLearning/releases/download/trainer-v0.2.0/otterlogic-trainer-0.2.0-win-x64.zip",
            Sha256 = "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789",
            Bytes = 123_456_789,
        };

        string json = bundle.ToJson();
        Assert.Contains("\"version\": \"0.2.0\"", json);
        Assert.Contains("\"url\"", json);
        Assert.Contains("\"sha256\"", json);
        Assert.Contains("\"bytes\": 123456789", json);

        Assert.Equal(bundle, TrainerBundle.FromJson(json));
    }

    [Fact]
    public void TheManifestIsReadAsTheBuildScriptWritesIt()
    {
        // Lower-case hash and no whitespace, as a script would emit it.
        const string json = """{"version":"0.2.0","url":"https://example.invalid/t.zip","sha256":"00ff","bytes":42}""";

        var bundle = TrainerBundle.FromJson(json);

        Assert.Equal("0.2.0", bundle.Version);
        Assert.Equal("00ff", bundle.Sha256);
        Assert.Equal(42, bundle.Bytes);
    }

    [Theory]
    [InlineData("""{"version": "0.2.0", "url": "https://x/y.zip"}""")]
    [InlineData("""{"url": "https://x/y.zip", "sha256": "ab"}""")]
    [InlineData("not json")]
    [InlineData("null")]
    public void AManifestMissingSomethingIsRefused(string json)
    {
        Assert.Throws<InvalidOperationException>(() => TrainerBundle.FromJson(json));
    }
}
